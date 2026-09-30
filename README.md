# Phy-HyDE
Phy-HyDE: Incorporating Physical Priors into Hypernetworks for Dynamic Generative Guidance

This repo currently holds the data and baseline-model code that Phy-HyDE builds on. It has two independent parts:

| Part | Files | Domain | What it does |
|---|---|---|---|
| A. Fashion CFG baseline | `fashion_cfg_diffusion.py`, `make_captions.py` | DeepFashion images | Class-conditional DDPM with classifier-free guidance (CFG), plus evaluation (FID, classifier accuracy, precision/recall) and a caption generator |
| B. Tahoe single-cell pipeline | `Tahoe_subset_to_Google_Drive.ipynb`, `Tahoe_PyTorch_train_validation.ipynb` | Tahoe-100M perturbation transcriptomics | Downloads a balanced subset, builds train/validation loaders, and reduces expression to 50 PCs for diffusion/flow models |

No datasets, checkpoints or outputs are committed. You supply the data as described below.

---

## Part A: DeepFashion CFG diffusion

### Setup

Requires Python >= 3.10. The script carries inline dependency metadata (PEP 723), so the easiest way to run it is [uv](https://docs.astral.sh/uv/):

```bash
uv run --script fashion_cfg_diffusion.py <command> [options]
```

uv installs `torch>=2.5`, `torchvision>=0.20`, `numpy`, `Pillow`, `torchmetrics` and `torch-fidelity` into an isolated environment. Without uv, `pip install` those packages and use `python fashion_cfg_diffusion.py ...`. A CUDA GPU is strongly recommended; `--device auto` falls back to CPU, which is much slower.

Dataset: download [DeepFashion Category and Attribute Prediction](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) (the `img/`, `Anno_coarse/`, `Anno_fine/` and `Eval/` folders) into one directory, referred to below as `$DATA`. The script reads:

- `Anno_coarse/list_category_cloth.txt`, `list_category_img.txt`, `list_bbox.txt`
- `Eval/list_eval_partition.txt` (train / val / test split)

### Usage

```bash
# 1. Sanity-check the data; prints split sizes and the zero-based class IDs
uv run --script fashion_cfg_diffusion.py check-data --data $DATA

# 2. Train the diffusion model (resumable with --resume)
uv run --script fashion_cfg_diffusion.py train --data $DATA --out runs/fashion --size 64 --epochs 50

# 3. Sample a grid of images for one class with guidance scale 3
uv run --script fashion_cfg_diffusion.py sample --out runs/fashion --class-id 2 --guidance 3.0

# 4. FID against real images (overall, or --fid-scope class --class-id N)
uv run --script fashion_cfg_diffusion.py evaluate --data $DATA --out runs/fashion

# 5. Train a ResNet-18 classifier, then run the guidance experiments
uv run --script fashion_cfg_diffusion.py train-classifier --data $DATA --out runs/fashion
uv run --script fashion_cfg_diffusion.py evaluate-experiments --data $DATA --out runs/fashion
```

Useful flags: `--max-train-images N` for a class-balanced subset on limited compute, `--max-steps N` for a quick smoke test, `--patience`/`--min-delta` for early stopping, `--guidance-scales 0 1 3 5` for `evaluate-experiments`. Run any command with `-h` for the full list.

### Outputs (in `--out`)

| File | From | Contents |
|---|---|---|
| `latest.pt`, `best.pt` | `train` | Checkpoints (latest epoch; lowest validation MSE) |
| `classes.json`, `history.jsonl` | `train` | Category names; per-epoch train/val MSE |
| `sample_class{N}_guidance{G}.png` | `sample` | Image grid |
| `fid_{overall\|classN}_{split}.json` | `evaluate` | FID report |
| `classifier_best.pt` | `train-classifier` | Best ResNet-18 by validation top-1 |
| `experiments_{split}.json` | `evaluate-experiments` | Top-1/top-5, precision/recall per guidance scale, per-class accuracy |

### How the code works (`fashion_cfg_diffusion.py`)

- **`FashionData`**: loads one split. Each image is cropped to its bounding box, padded to a white square at `--size`, optionally flipped (train only), and scaled to [-1, 1]. Labels are converted from one-based to zero-based. `--max-*-images` gives a deterministic, class-balanced subset.
- **`UNet`**: a small conditional U-Net (two down/up levels, residual blocks with GroupNorm, one self-attention block in the middle). The conditioning vector is a sinusoidal timestep embedding plus a learned class embedding. The embedding table has one extra row, the **null label**, used for unconditional prediction.
- **`Diffusion`**: 1000-step cosine noise schedule. `noise()` applies the forward process; the network is trained to predict the added noise (MSE).
- **Classifier-free guidance**: during `train`, each label is replaced by the null label with probability `--drop-probability` (default 0.1). During sampling, `generate_images` runs a deterministic DDIM-style sampler (`--sample-steps`, default 50) and combines the two predictions as `uncond + guidance * (cond - uncond)`. Guidance 0 is fully unconditional, 1 is plain conditional, >1 sharpens class adherence.
- **`train`**: AdamW, mixed precision on CUDA, gradient clipping, per-epoch validation MSE with fixed noise, atomic checkpoints, and early stopping. Resuming checks that the config matches.
- **`evaluate`**: FID (2048-d Inception features, via torchmetrics). For `overall` scope the generated class frequencies mirror the real reference set. Small sample sizes are flagged in the report.
- **`train-classifier` / `evaluate-experiments`**: a ResNet-18 (ImageNet-pretrained by default) on the same crops. For each guidance scale, generated images are scored by classifier top-1/top-5 accuracy (does the image match its requested class?) and by k-NN precision/recall in classifier feature space (fidelity vs. diversity). It can also load a notebook-trained checkpoint from `$DATA/fashion_resnet18_classifier`.

### Caption generator (`make_captions.py`)

Builds `captions.csv` (`image_path,caption`), one sentence per DeepFashion image, e.g. "This is a floral chiffon dress with a v-neckline and short sleeves." Run it from the DeepFashion root (it uses `ROOT = "."`) with `numpy` installed:

```bash
cd $DATA && python /path/to/make_captions.py
```

It combines the category, the 1000 noisy coarse attributes (all 289k images) and the 26 cleaner fine attributes (20k images; these take priority where they overlap). Only positive attributes are used. A hand-written vocabulary table maps each attribute to a slot (style, fit, colour, fabric, pattern, neckline, etc.) with a compatibility rule so it only applies to sensible garments (e.g. "maxi" only for dresses and skirts, "skinny" only for trousers). `build_sentence` then resolves contradictions, caps each slot, and assembles the text.

---

## Part B: Tahoe-100M single-cell pipeline (Google Colab)

Both notebooks are designed for [Google Colab](https://colab.research.google.com/) with Google Drive mounted. Upload each `.ipynb` via **File → Upload notebook** and run the cells in order.

### 1. `Tahoe_subset_to_Google_Drive.ipynb`: build the subset

A CPU runtime is enough. It creates a balanced subset of [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) in `MyDrive/Tahoe/subsets/<SUBSET_NAME>`.

- **Default subset** (`cfg_25lines_100drugs`): 25 cell lines x 100 drugs x 3 doses (0.05, 0.5, 5 µM), up to 100 cells per treated line/sample and 200 per DMSO control sample (max about 1.0M cells). It spans 13 organs and all 26 annotated mechanisms of action.
- **Workflow**: install libraries, mount Drive, set parameters (section 3), prepare the plan (saves metadata and `panel_review.json`), select cells (section 6), download expression (section 7), then verify coverage (section 8).
- **Selection** is a hash-ranked random sample per line/sample group with QC filters (`pass_filter == "full"`, at least 200 genes, mitochondrial fraction <= 0.20). Controls are the exact label `DMSO_TF`.
- **Output**: `cells/part-XXXXX.parquet`, each row holding sparse gene-token (uint16) and expression (float32) lists plus cell line, drug, dose, plate, sample and cell ID.
- **Resumable**: after a disconnect, rerun sections 1-5 with the same settings, then sections 6-7. Use a new `SUBSET_NAME` when changing the line/drug counts or seed.

### 2. `Tahoe_PyTorch_train_validation.ipynb`: loaders and PCA features

Reads the subset from Drive (set `SUBSET_NAME` to match; if you downloaded the zip instead, the local-loading cell unzips it to `/content/data`).

**Sections 1-6, raw sparse loaders**
- **Split**: holds out about 20% of observed (cell line, drug) pairs for validation; all cells, doses and replicates of a held-out pair stay together, and each held-out drug and cell line still appears in training. DMSO cells are split individually by a stable cell-ID hash. The compact index is written once to `splits/<SPLIT_NAME>/` and reused.
- **`TahoeCells`** lazily reads one Parquet row group at a time (2 cached); **`FragmentBatches`** groups each batch by row group to limit Drive reads; `collate_raw_four` yields batches with exactly `genes`, `expressions`, `drug`, `cell_line_id` (ragged sparse tensors and string labels). Keep `NUM_WORKERS=0` on mounted Drive.

**Section 7, dense PC features for diffusion / flow models**
Everything is fit on training cells only:
1. Library-size normalise to 10,000 and `log1p`.
2. Select the top 2,000 highly variable genes (Seurat-style binned dispersion).
3. Z-score and clip at +/-10.
4. Streaming PCA to 50 components (exact, from sum and Gram-matrix moments).
5. Per treated (cell line, drug) pair, k-means (k = 3) and order clusters by distance to that line's DMSO centroid, so cluster 0 is closest to control and 2 is farthest. DMSO cells are labelled -1.

`TahoePCs` serves batches with `pcs` (B x 50, scaled), `drug`, `cell_line_id`, `cluster`, `is_control`. `pcs_to_hvg_expression` maps samples back to the 2,000-gene space (lossy). Results are saved next to the split index with `pca_manifest.json` and can be downloaded as a zip (cells 38 and 48-49) to reuse in another notebook. Section 7f (checks and plots) is optional. Set `MAX_BATCHES` for a quick partial trial that writes to a separate `_smoke` folder.

