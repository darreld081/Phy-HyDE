"""Tahoe-100M cell states: 25 cell lines x 100 drugs (+ DMSO control), 50 principal components.

The data folder is the output of notebooks/tahoe/tahoe_colab_downloader_with_pca.ipynb:
library-size normalisation and log1p, 2000 highly variable genes, z-scoring and clipping at
+-10, exact PCA to 50 components, all fitted on the training split only. The split holds out
about 20% of (cell line, drug) PAIRS: every cell of a held-out pair is in validation, while
each of its cell line and drug appears in some training pair. DMSO cells are split by a hash
so both splits have controls. `cluster` is k-means (k = 3) per treated pair on its raw PCs,
ordered by distance to that line's DMSO centroid (0 least perturbed, 2 most; -1 for DMSO).

`TahoePCs`, `make_pca_loaders`, `load_pca_model` and `pcs_to_hvg_expression` are the loader
from notebooks/tahoe/Tahoe_dataloader_example.ipynb, unchanged apart from resolving the
folder through `find_pca_dir`. `make_loaders` below adapts it to the notebooks' contract:
loaders yielding `(x, cond_id)` and a `meta` dict with `n_conditions`, `shape`, `cond_text`.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

SEED = 42
SUBSET = "25lines_100drugs_pca_data_v2/content/data/Tahoe/subsets/cfg_25lines_100drugs"
PCA_REL = "splits/pair_seed42_val20_raw4/pca50_hvg2000"

# Places the pca50_hvg2000 folder might be, relative to the notebook's folder or its parents:
# the zip unpacked as is (content/data/Tahoe/...), the zip unpacked into a folder named after it
# (data/25lines_100drugs_pca_data_v2/content/data/Tahoe/...), or just the Tahoe tree
# (data/Tahoe/...). Then Colab session storage and Drive. TAHOE_PCA_DIR overrides all of it.
_TAHOE = f"Tahoe/subsets/cfg_25lines_100drugs/{PCA_REL}"
SEARCH_PATHS = tuple(
    f"{up}/{layout}" for up in (".", "..", "../..", "../../..")
    for layout in (f"content/data/{_TAHOE}", f"data/{SUBSET}/{PCA_REL}", f"data/{_TAHOE}", f"{SUBSET}/{PCA_REL}")
) + (
    f"/content/content/data/{_TAHOE}", f"/content/data/{_TAHOE}",
    f"/content/drive/MyDrive/content/data/{_TAHOE}", f"/content/drive/MyDrive/data/{_TAHOE}",
    f"/content/drive/MyDrive/{SUBSET}/{PCA_REL}",
)


def find_pca_dir(root=None):
    """Return the folder holding pca_model.npz and pcs_*.npy, or raise with the paths tried.
    The environment variable TAHOE_PCA_DIR, if set, is tried first."""
    candidates = [root] if root is not None else [os.environ.get("TAHOE_PCA_DIR"), *SEARCH_PATHS]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "pca_model.npz")):
            return Path(path)
    raise FileNotFoundError("no Tahoe PCA folder found; tried " + ", ".join(
        [str(root)] if root is not None else SEARCH_PATHS))


# ---- the provided loader, as in Tahoe_dataloader_example.ipynb ----------------------------

def make_hvg_col(hvg_ids):
    hvg_col = np.full(int(hvg_ids.max()) + 1, -1, dtype=np.int64)
    hvg_col[hvg_ids] = np.arange(len(hvg_ids))
    return hvg_col


def load_pca_model(pca_dir=None):
    pca_dir = find_pca_dir(pca_dir)
    with np.load(Path(pca_dir) / "pca_model.npz") as saved:
        model = {key: saved[key] for key in saved.files}
    model["pc_scale"] = float(model["pc_scale"])
    model["hvg_col"] = make_hvg_col(model["hvg_ids"])
    return model


class TahoePCs(Dataset):
    """Self-contained: only needs the files under `pca_dir` (pca_model.npz, pcs_*.npy, cluster_*.npy,
    line_id_*.npy, drug_id_*.npy, dose_*.npy, pca_manifest.json). Copy that folder anywhere and this class works
    with no raw Parquet loader and no Drive mount."""

    def __init__(self, split, pca_dir=None, pc_scale=None):
        pca_dir = find_pca_dir(pca_dir)
        labels = json.loads((pca_dir / "pca_manifest.json").read_text())
        pcs = np.load(pca_dir / f"pcs_{split}.npy")
        keep = np.isfinite(pcs).all(axis=1)      # all rows unless this was a MAX_BATCHES smoke run
        self.pc_scale = float(np.load(pca_dir / "pca_model.npz")["pc_scale"]) if pc_scale is None else pc_scale
        self.pcs = torch.from_numpy(pcs[keep] / self.pc_scale).float()
        self.cluster = torch.from_numpy(np.load(pca_dir / f"cluster_{split}.npy")[keep].astype(np.int64))
        self.drug_id = np.load(pca_dir / f"drug_id_{split}.npy")[keep].astype(np.int64)
        self.line_id = np.load(pca_dir / f"line_id_{split}.npy")[keep].astype(np.int64)
        self.dose = torch.from_numpy(np.load(pca_dir / f"dose_{split}.npy")[keep].astype(np.float32))  # dose in uM
        self.drug_names = np.asarray(labels["drug_names"])
        self.line_names = np.asarray(labels["line_names"])
        self.is_control = torch.from_numpy(self.drug_id == 0)

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, index):
        index = np.asarray(index)
        return {"pcs": self.pcs[index], "drug": self.drug_names[self.drug_id[index]].tolist(),
                "cell_line_id": self.line_names[self.line_id[index]].tolist(),
                "dose": self.dose[index],
                "cluster": self.cluster[index], "is_control": self.is_control[index]}


def make_pca_loaders(batch_size=256, seed=SEED, pca_dir=None):
    train_ds = TahoePCs("train", pca_dir)
    val_ds = TahoePCs("validation", pca_dir, pc_scale=train_ds.pc_scale)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds, batch_size=None, num_workers=0,
                       sampler=BatchSampler(RandomSampler(train_ds, generator=generator), batch_size, drop_last=True))
    validation = DataLoader(val_ds, batch_size=None, num_workers=0,
                            sampler=BatchSampler(SequentialSampler(val_ds), batch_size, drop_last=False))
    return train, validation


def pcs_to_hvg_expression(pcs, model=None):
    """Scaled PCs (as served by the loaders) -> log-normalised HVG expression, shape (..., N_HVG)."""
    model = load_pca_model() if model is None else model
    z = (np.asarray(pcs, dtype=np.float64) * model["pc_scale"]) @ model["components"] + model["pca_mean"]
    return z * model["scaler_std"] + model["scaler_mean"]


# ---- adapter to the notebooks' (x, cond_id) contract ---------------------------------------

def condition_texts(pca_dir=None):
    """Drug and cell-line descriptions from panel_review.json: the Tahoe drug metadata
    (mechanism of action, targets) and cell-line metadata (name, organ). Written from the
    public metadata, not from any result."""
    review = json.loads((find_pca_dir(pca_dir).parents[2] / "panel_review.json").read_text(encoding="utf-8"))
    drug = {"DMSO_TF": "DMSO vehicle control, no drug"}
    for d in review["drugs"]:
        text = d["drug"]
        if d.get("moa-fine") and d["moa-fine"] != "unclear":
            text += f", {d['moa-fine'].lower()}"
        if d.get("targets"):
            text += f" targeting {d['targets']}"
        drug[d["drug"]] = text
    line = {}
    for l in review["cell_lines"]:
        name = l["names"][0] if l.get("names") else l["cell_line_id"]
        organ = l["organs"][0].lower() if l.get("organs") else "unknown tissue"
        line[l["cell_line_id"]] = f"{name} {organ} cells"
    return drug, line


class _Pairs(Dataset):
    def __init__(self, x, cond):
        self.x, self.cond = x, cond

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.cond[i]


def make_loaders(root=None, batch_size=256, seed=SEED, condition="drug", dose=None, limit=0):
    """`(train_loader, val_loader, meta)`; loaders yield `(x, cond_id)` with x of shape (B, 50).

    condition: what `cond_id` indexes.
        "drug"    101 ids: 0 is DMSO, 1..100 the drugs. Text = drug, mechanism, targets.
        "pair"    one id per (cell line, drug) pair seen in either split; text adds the line.
                  Validation pairs never occur in training, so validation loss here measures
                  generalisation to unseen line-drug combinations.
        "cluster" 4 ids: 0 control, 1..3 the per-pair k-means clusters from least to most
                  perturbed. The k-means objective for TFG-style guidance.
    dose: None keeps every dose; a number keeps that dose (in uM) plus the controls.
    limit > 0 truncates both splits, for smoke tests. `seed` only seeds the shuffling.
    """
    pca_dir = find_pca_dir(root)
    train_ds = TahoePCs("train", pca_dir)
    val_ds = TahoePCs("validation", pca_dir, pc_scale=train_ds.pc_scale)
    drug_text, line_text = condition_texts(pca_dir)

    if condition == "drug":
        n_cond = len(train_ds.drug_names)
        cond_text = [drug_text[d] for d in train_ds.drug_names]
        ids = lambda ds: torch.from_numpy(ds.drug_id)
    elif condition == "pair":
        pairs = sorted({(int(l), int(d)) for ds in (train_ds, val_ds)
                        for l, d in zip(ds.line_id, ds.drug_id)})
        index = {p: i for i, p in enumerate(pairs)}
        n_cond = len(pairs)
        cond_text = [f"{drug_text[train_ds.drug_names[d]]}, in {line_text[train_ds.line_names[l]]}"
                     for l, d in pairs]
        ids = lambda ds: torch.tensor([index[(int(l), int(d))] for l, d in zip(ds.line_id, ds.drug_id)])
    elif condition == "cluster":
        n_cond = 4
        cond_text = ["DMSO vehicle control, unperturbed cells",
                     "least perturbed treated cells, expression close to control",
                     "moderately perturbed treated cells",
                     "most perturbed treated cells, expression far from control"]
        ids = lambda ds: ds.cluster + 1
    else:
        raise ValueError(f"condition must be drug, pair or cluster, got {condition!r}")

    def select(ds):
        keep = torch.ones(len(ds), dtype=torch.bool)
        if dose is not None:
            keep &= (ds.dose == float(dose)) | ds.is_control
        idx = keep.nonzero().squeeze(1)
        if limit:
            idx = idx[:limit]
        return ds.pcs[idx], ids(ds)[idx].long(), idx

    x_tr, c_tr, i_tr = select(train_ds)
    x_va, c_va, i_va = select(val_ds)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(_Pairs(x_tr, c_tr), batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(_Pairs(x_va, c_va), batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_pca_model(pca_dir)
    # (drug_id, line_id) per condition id, so a backbone with separate drug and line embeddings
    # can be driven by one cond_id. Null ids are n_drugs and n_lines. For "drug" the line is null.
    n_drugs, n_lines = len(train_ds.drug_names), len(train_ds.line_names)
    if condition == "pair":
        cond_ids_table = torch.tensor([[d, l] for l, d in pairs], dtype=torch.long)
    elif condition == "drug":
        cond_ids_table = torch.stack([torch.arange(n_cond), torch.full((n_cond,), n_lines)], 1)
    else:
        cond_ids_table = torch.full((n_cond, 2), -1, dtype=torch.long)   # clusters map to no (drug, line)

    meta = {
        "n_conditions": n_cond,
        "shape": (x_tr.shape[1],),
        "cond_text": cond_text,
        "condition": condition,
        "n_drugs": n_drugs, "n_lines": n_lines,
        "cond_ids_table": cond_ids_table,        # (n_conditions, 2): drug_id, line_id
        "dose_filter": dose,
        "n_train": int(len(x_tr)), "n_val": int(len(x_va)),
        "drug_names": train_ds.drug_names.tolist(),
        "line_names": train_ds.line_names.tolist(),
        "doses_uM": sorted(set(train_ds.dose.tolist())),
        "pc_scale": train_ds.pc_scale,
        # per-line DMSO centroid in the same scaled PC space as x; x - dmso_reference[line]
        # is the perturbation shift
        "dmso_reference": torch.from_numpy(model["dmso_reference"] / train_ds.pc_scale).float(),
        "train_line_id": torch.from_numpy(train_ds.line_id)[i_tr],
        "val_line_id": torch.from_numpy(val_ds.line_id)[i_va],
        "train_drug_id": torch.from_numpy(train_ds.drug_id)[i_tr],
        "val_drug_id": torch.from_numpy(val_ds.drug_id)[i_va],
        "train_cluster": train_ds.cluster[i_tr], "val_cluster": val_ds.cluster[i_va],
        "split": "held-out (cell line, drug) pairs, seed 42, ~20%; DMSO split by cell hash",
    }
    return train_loader, val_loader, meta


if __name__ == "__main__":
    for cond in ("drug", "pair", "cluster"):
        tr, va, meta = make_loaders(condition=cond, batch_size=512)
        x, c = next(iter(tr))
        print(f"{cond:8s} conditions {meta['n_conditions']:5d} | train {meta['n_train']:,} val {meta['n_val']:,} "
              f"| x {tuple(x.shape)} cond {tuple(c.shape)} max id {int(c.max())}")
        print("         ", meta["cond_text"][1][:90])
    tr, va, meta = make_loaders(condition="pair")
    train_pairs = set(zip(meta["train_line_id"].tolist(), meta["train_drug_id"].tolist()))
    val_pairs = set(zip(meta["val_line_id"].tolist(), meta["val_drug_id"].tolist()))
    treated_overlap = {p for p in train_pairs & val_pairs if p[1] != 0}
    print("treated pairs: train", len({p for p in train_pairs if p[1]}), "val", len({p for p in val_pairs if p[1]}),
          "| overlap", len(treated_overlap), "(must be 0)")
    assert not treated_overlap
    xs = tr.dataset.x
    print("x per-dim std (first 5):", xs[:, :5].std(0).numpy().round(2), "| overall rms", round(float(xs.pow(2).mean().sqrt()), 3))
    tab = meta["cond_ids_table"]
    assert tab.shape == (meta["n_conditions"], 2) and int(tab[:, 0].max()) < meta["n_drugs"] and int(tab[:, 1].max()) < meta["n_lines"]
    print("cond_ids_table", tuple(tab.shape), "| treated pairs in table:", int((tab[:, 0] != 0).sum()))
    tr, va, meta = make_loaders(condition="drug", dose=5.0, limit=1000)
    print("dose 5 uM + controls, limit 1000: train", meta["n_train"], "val", meta["n_val"])
    print("ok")
