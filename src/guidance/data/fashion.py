"""DeepFashion (Category and Attribute Prediction benchmark) at 64x64, with per-image captions.

`FashionDataset` and `read_rows` are the loader from a teammate's Modality-1 notebooks
(CIS6720_Proj1_DiffusionCode_Modality_1.ipynb and the flow-matching twin), kept as written:
crop to the garment box, pad to a white square, resize, optional flip, pixels in [-1, 1], the
official train/val partition, a class-balanced validation subset. Liu et al., CVPR 2016.

`make_loaders` adapts it to the notebooks' contract: loaders yield `(x, idx)` where `idx` is the
image's position in its split, and `meta` carries, per position, the class label and the caption
from captions.csv, so a notebook can condition the backbone on the label (`meta["labels_train"]`)
and a text method on the caption (`meta["captions_train"]`). The caption, e.g. "This is a chic
sleeveless pleated blouse with a v-neckline.", carries attributes the 50-way label does not.

Expected folder: <root>/img/..., <root>/Anno_coarse/{list_category_cloth,list_category_img,
list_bbox}.txt, <root>/Eval/list_eval_partition.txt, <root>/captions.csv.
"""

import csv
import glob
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

SEARCH_PATHS = tuple(f"{up}/{name}" for up in (".", "..", "../..", "../../..") for name in ("DeepFashion", "data/DeepFashion", "data/fashion")) + (
    "/content/DeepFashion", "/content/drive/MyDrive/DeepFashion")
MARKERS = ("Anno_coarse", "Eval/list_eval_partition.txt")


def _is_root(path):
    return bool(path) and all(os.path.exists(os.path.join(path, p)) for p in MARKERS)


def find_captions(root):
    """captions.csv next to the annotations, or up to two folders above (the zip ships it separately),
    or wherever FASHION_CAPTIONS points."""
    root = Path(root)
    for p in [os.environ.get("FASHION_CAPTIONS"), root / "captions.csv", root.parent / "captions.csv",
              root.parent.parent / "captions.csv"]:
        if p and os.path.exists(p):
            return Path(p)
    raise FileNotFoundError(f"no captions.csv in {root}, its parents, or FASHION_CAPTIONS")


ZIP_DIRS = (".", "./data", "..", "../data", "../..", "../../data", "../../..", "../../../data")


def _candidates(root):
    candidates = [root] if root is not None else [os.environ.get("FASHION_ROOT"), *SEARCH_PATHS]
    if root is None:
        # any folder next to the notebook, then up to two levels under data/ from here or the parents
        candidates += sorted(glob.glob("./*/")) + sorted(glob.glob("./*/*/"))
        for up in (".", "..", "../..", "../../.."):
            candidates += sorted(glob.glob(f"{up}/data/*/")) + sorted(glob.glob(f"{up}/data/*/*/"))
    return [c for c in candidates if c]


def _find_zips():
    """(annotation zip, image zip) found next to the notebook or under data/, either may be None.
    The annotation zip contains Eval/list_eval_partition.txt; the image zip's entries live under img/."""
    import zipfile
    anno = imgs = None
    for d in ZIP_DIRS:
        for z in sorted(glob.glob(f"{d}/*.zip")):
            try:
                names = zipfile.ZipFile(z).namelist()
            except zipfile.BadZipFile:
                continue
            if anno is None and any(n.endswith("Eval/list_eval_partition.txt") for n in names):
                anno = z
            elif imgs is None and names and all(n.startswith("img/") for n in names[:50]):
                imgs = z
    return anno, imgs


def _unzip(path, dest, log):
    import time, zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        log(f"unzipping {path} ({len(names):,} entries) into {dest} ...")
        t0 = time.time()
        for i, n in enumerate(names, 1):
            z.extract(n, dest)
            if i % 50_000 == 0:
                log(f"  {i:,} / {len(names):,} ({time.time() - t0:.0f}s)")
        log(f"  done in {time.time() - t0:.0f}s")


def find_root(root=None, log=print):
    """Return the folder holding Anno_coarse/, Eval/list_eval_partition.txt and img/ (the unzipped
    'Category and Attribute Prediction Benchmark'), or raise with the paths tried.

    Tried in order: the argument, FASHION_ROOT, the usual names, any folder next to the notebook
    (up to two levels), then any folder up to two levels under data/ (from the notebook folder or
    its parents). If nothing is unpacked yet but the zips are next to the notebook or under data/,
    they are unzipped here: the annotation zip into ./DeepFashion/, the image zip into the folder
    that holds the annotations (several minutes, once). captions.csv may sit inside the folder or
    up to two folders above."""
    found = next((Path(p) for p in _candidates(root) if _is_root(p)), None)
    if found is None and root is None:
        anno, _ = _find_zips()
        if anno is not None:
            _unzip(anno, Path(anno).parent / "DeepFashion", log)
            found = next((Path(p) for p in _candidates(None) if _is_root(p)), None)
    if found is None:
        raise FileNotFoundError("no DeepFashion folder (or zip) found; tried " + ", ".join(_candidates(root)))
    if not (found / "img").is_dir():
        _, imgs = _find_zips()
        if imgs is None:
            raise FileNotFoundError(f"{found} has the annotations but no img/ folder and no image zip is nearby: "
                                    f"unzip the image archive into it")
        _unzip(imgs, found, log)
    find_captions(found)
    return found


# ---- the provided loader, as in the Modality-1 notebooks ---------------------------------

def read_rows(path):
    with Path(path).open(encoding="utf-8") as file:
        next(file)  # Number of records.
        next(file)  # Column names.
        for line in file:
            if line.strip():
                yield line.split()


class FashionDataset(Dataset):
    def __init__(self, root, split, captions, image_size=64, flip=False, limit=0, seed=42):
        self.root = Path(root)
        self.captions = captions
        self.image_size = image_size
        self.flip = flip
        annotation = self.root / "Anno_coarse"

        self.classes = [" ".join(row[:-1]) for row in read_rows(annotation / "list_category_cloth.txt")]
        labels = {row[0]: int(row[1]) - 1 for row in read_rows(annotation / "list_category_img.txt")}
        boxes = {row[0]: tuple(map(int, row[1:5])) for row in read_rows(annotation / "list_bbox.txt")}
        self.records = [(path, labels[path], boxes[path])
                        for path, partition in read_rows(self.root / "Eval/list_eval_partition.txt")
                        if partition == split and path in labels and path in boxes]
        if not self.records:
            raise ValueError(f"No {split} images found under {self.root}")

        if limit and limit < len(self.records):
            # The small validation set stays balanced across the available classes.
            groups = defaultdict(list)
            for record in self.records:
                groups[record[1]].append(record)
            rng = random.Random(seed)
            for group in groups.values():
                rng.shuffle(group)
            selected = []
            class_ids = sorted(groups)
            while len(selected) < limit and class_ids:
                for class_id in class_ids[:]:
                    if len(selected) == limit:
                        break
                    if groups[class_id]:
                        selected.append(groups[class_id].pop())
                    else:
                        class_ids.remove(class_id)
            self.records = selected

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative_path, label, (x1, y1, x2, y2) = self.records[index]
        with Image.open(self.root / relative_path) as source:
            image = source.convert("RGB")
            x1, x2 = max(0, x1), min(image.width, x2)
            y1, y2 = max(0, y1), min(image.height, y2)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
            image = ImageOps.pad(image, (self.image_size, self.image_size),
                                 method=Image.Resampling.BICUBIC, color=(255, 255, 255))
            if self.flip and random.random() < 0.5:
                image = ImageOps.mirror(image)
            pixels = np.asarray(image, dtype=np.uint8).copy()
        # Convert RGB pixels from [0, 255] to [-1, 1].
        x0 = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1)
        return x0, label, self.captions[relative_path]


def load_captions(root):
    captions_path = find_captions(root)
    with captions_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_path", "caption"]:
            raise ValueError(f"Unexpected caption columns: {reader.fieldnames}")
        return {row["image_path"]: row["caption"] for row in reader}


# ---- adapter to the notebooks' (x, idx) contract ------------------------------------------

class _Indexed(Dataset):
    """Yields (image, position) so the notebook can look up label and caption by position."""

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x, _, _ = self.inner[i]
        return x, i


def make_loaders(root=None, image_size=64, batch_size=64, val_limit=2048, seed=42, limit=0, flip=True,
                 num_workers=2):
    """`(train_loader, val_loader, meta)`; loaders yield `(x, idx)` with x of shape (3, size, size).

    val_limit: size of the class-balanced validation subset (the notebooks' 2048).
    limit > 0 truncates the training set, for smoke tests. `seed` drives the validation subset and
    the shuffling; it does not touch model initialisation.
    """
    root = find_root(root)
    captions = load_captions(root)
    num_workers = int(os.environ.get("FASHION_NUM_WORKERS", num_workers))   # 0 on Windows or in a script
    # limit > 0: FashionDataset's own class-balanced, seeded subset (the same rule the teammates'
    # script applies to --max-train-images), so a small run still sees every class.
    train_ds = FashionDataset(root, "train", captions, image_size, flip=flip, limit=limit, seed=seed)
    val_ds = FashionDataset(root, "val", captions, image_size, limit=val_limit, seed=seed)

    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(_Indexed(train_ds), batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=g, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(_Indexed(val_ds), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    n_classes = len(train_ds.classes)
    active = sorted({r[1] for r in train_ds.records})
    meta = {
        "n_conditions": n_classes,                               # class labels; the null token is n_classes
        "n_classes": n_classes,
        "shape": (3, image_size, image_size),
        "class_names": train_ds.classes,
        "active_class_ids": active,                              # 46 of the 50 have images
        "cond_text": [f"a photo of a {c.lower().replace('_', ' ')}" for c in train_ds.classes],
        "labels_train": torch.tensor([r[1] for r in train_ds.records]),
        "labels_val": torch.tensor([r[1] for r in val_ds.records]),
        "captions_train": [captions[r[0]] for r in train_ds.records],
        "captions_val": [captions[r[0]] for r in val_ds.records],
        "paths_val": [r[0] for r in val_ds.records],
        "n_train": len(train_ds), "n_val": len(val_ds),
        "root": str(root),
        "split": "DeepFashion official train / val partition; validation is a class-balanced subset",
    }
    return train_loader, val_loader, meta


if __name__ == "__main__":
    tr, va, meta = make_loaders(num_workers=0, batch_size=8)
    x, idx = next(iter(tr))
    print("x", tuple(x.shape), f"[{float(x.min()):.2f}, {float(x.max()):.2f}]", "| idx", tuple(idx.shape))
    print(f"train {meta['n_train']:,} | val {meta['n_val']:,} | classes {meta['n_classes']} "
          f"(active {len(meta['active_class_ids'])}) | root {meta['root']}")
    i = int(idx[0])
    print("first item: class", meta["class_names"][int(meta["labels_train"][i])], "| caption:", meta["captions_train"][i])
    assert len(meta["captions_train"]) == meta["n_train"] and len(meta["labels_val"]) == meta["n_val"]
    print("ok")
