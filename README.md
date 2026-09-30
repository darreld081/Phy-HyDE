# Phy-HyDE
Phy-HyDE: Incorporating Physical Priors into Hypernetworks for Dynamic Generative Guidance

This repo currently holds the data and baseline-model code that Phy-HyDE builds on. It has two independent parts:

| Part | Files | Domain | What it does |
|---|---|---|---|
| A. Fashion CFG baseline | `fashion_cfg_diffusion.py`, `make_captions.py` | DeepFashion images | Class-conditional DDPM with classifier-free guidance (CFG), plus evaluation (FID, classifier accuracy, precision/recall) and a caption generator |
| B. Tahoe single-cell pipeline | `Tahoe_subset_to_Google_Drive.ipynb`, `Tahoe_PyTorch_train_validation.ipynb` | Tahoe-100M perturbation transcriptomics | Downloads a balanced subset, builds train/validation loaders, and reduces expression to 50 PCs for diffusion/flow models |
| C. Comparison methods | `notebooks/recent_methods/`, `src/guidance/` | both | TC-LoRA, text cross-attention and training-free guidance (TFG) on the Part A and Part B base models, one notebook per method and modality, with a shared evaluation and a manifest per run |

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

---

## Part C: comparison methods

This part adds the comparison methods: TC-LoRA, text cross-attention and training-free guidance
(TFG). There is one notebook per method and modality. They use the base models from Parts A and
B and import shared code from `src/guidance/`. The first cell of each notebook adds `src/` to the
path when run inside a checkout, and clones this repo when run on Colab.

```
notebooks/recent_methods/
  tahoe/    cfg_cells.ipynb   tc_lora_cells.ipynb   text_xattn_cells.ipynb   tfg_cells.ipynb
  fashion/  cfg_fashion.ipynb tc_lora_fashion.ipynb text_xattn_fashion.ipynb tfg_fashion.ipynb
src/guidance/
  baselines/  tc_lora.py  cross_attention.py  tfg.py
  data/       cells.py  fashion.py
  eval/       cell_metrics.py  image_metrics.py
```

| Notebook | Method | Paper | Official code | What trains | Condition |
|---|---|---|---|---|---|
| `cfg_*` | label embedding + classifier-free guidance | Ho & Salimans, [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) (NeurIPS 2021 workshop) | none released | the base model, with the Part A / Part B recipe | class label (fashion) or (cell line, drug) pair (Tahoe) |
| `tc_lora_*` | TC-LoRA | Cho et al., [arXiv:2510.09561](https://arxiv.org/abs/2510.09561) (SpaVLE @ NeurIPS 2025) | none released | a hypernetwork that writes per-step LoRA factors for the frozen base | CLIP embedding of the caption / pair description |
| `text_xattn_*` | text cross-attention | PixArt-α, Chen et al., [arXiv:2310.00426](https://arxiv.org/abs/2310.00426) (ICLR 2024) | [github.com/PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) | one cross-attention block on the frozen base | CLIP token embeddings of the same text |
| `tfg_*` | training-free guidance | TFG, Ye et al., [arXiv:2409.15761](https://arxiv.org/abs/2409.15761) (NeurIPS 2024) | [github.com/YWolfeee/Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance) | nothing; a fixed classifier steers sampling | target class / drug |

### The four methods

**Label embedding with classifier-free guidance (`cfg_*`)**
Method: the base model itself. The condition enters through a learned embedding (the class label,
or the drug and cell-line ids) added to the timestep embedding. During training the embedding is
replaced by a blank label 10% of the time, so the same network can also predict without the
condition. Classifier-free guidance is the sampling step that uses both branches: run the network
with and without the condition and combine the two outputs with a weight w. w = 1 is plain
conditional sampling; larger w pushes harder toward the condition.
Why: this is how the Part A and Part B base models are conditioned, and CFG is the standard
guidance method, so this row is the reference for both questions: how the condition gets in, and
what extra guidance buys.
Ours: the recipes unchanged. For Tahoe the drug and cell line are blanked together, and the
evaluation uses held-out (cell line, drug) pairs at w = 0, 1, 3 and 6.

**TC-LoRA (`tc_lora_*`)**
Method: the diffusion model is frozen. A small second network, the hypernetwork, reads the
timestep and the condition and outputs LoRA weight updates for the frozen model. Only the
hypernetwork is trained, so the update can differ at every step and for every condition.
Why: this project builds a hypernetwork that controls a diffusion model, and TC-LoRA is the
closest published version of that idea.
Ours: written from the paper, since no code was released. The paper adapts attention layers in a
transformer and uses a depth map as the condition. We adapt the convolutions of the U-Net and the
linear layers of the MLP, use a CLIP embedding of the caption or the pair description as the
condition, and give the frozen model a blank label so the text is its only source of the
condition.

**Text cross-attention (`text_xattn_*`)**
Method: the diffusion model is frozen and one cross-attention block is added. The block lets the
model's features attend to the text tokens. Only the block is trained.
Why: cross-attention is how text-to-image models read text, so it is the standard
text-conditioning baseline.
Ours: PixArt-α has such a block in every layer and trains the whole model. We add one block, at
the U-Net bottleneck or after the MLP's input projection, use CLIP tokens instead of T5, and train
nothing else. Sampling uses no guidance.

**Training-free guidance (`tfg_*`)**
Method: nothing is trained. At each sampling step the model predicts the clean sample, a
classifier scores it for the target class, and the sample is nudged along the gradient of that
score.
Why: it is the training-free alternative: no conditioning is learned, and the classifier does all
the work at sampling.
Ours: TFG needs an unconditional model and a classifier. The base model run with a blank label
serves as the unconditional model; for Tahoe the cell line is still given and only the drug is
blank. The classifier is a ResNet fine-tuned on DeepFashion for images and, for Tahoe, a Gaussian
classifier per drug fitted on the training cells. The predicted clean sample is clamped to the
data range, as the paper does with [-1, 1] pixels. A different classifier scores the results.

### Running

Run `cfg_*` first. It trains the base model (the Part A recipe for fashion, the Part B model for
Tahoe) and saves `checkpoints/diffusion_fashion/best.pt` or `checkpoints/diffusion_cfg/best.pt`.
The other three notebooks of that modality load it, so all four methods sit on the same weights.
Run the notebooks of one modality from the same folder.

Data: the Tahoe notebooks read the PCA folder from Part B (unzipped next to the notebook, under
`data/` at the repo root, or the path in `TAHOE_PCA_DIR`). The fashion notebooks read the
DeepFashion folders from Part A plus `captions.csv` from `make_captions.py` (next to the notebook,
under `data/`, or `FASHION_ROOT` and `FASHION_CAPTIONS`); the zips are unpacked if only they are
present. The fine-tuned classifier in `models/` is picked up and reused.

Any setting can be changed from the environment without editing a cell: `CFG_<NAME>=value`, for
example `CFG_LIMIT=5000 CFG_BASE_EPOCHS=1` for a smoke run. Each notebook writes
`manifest_<name>.json` with the commit, config, hardware, training time and every reported number.

### How the notebooks work

Every notebook has the same sequence of cells:

- **Config**: one dataclass with every setting, printed first, so a saved run shows what it ran with.
- **Data**: `make_loaders` from `src/guidance/data`. Tahoe loaders yield `(x, pair_id)` for the 50 PCs and the (cell line, drug) pair, with the pair's text (drug mechanism and targets, cell line name and organ) in `meta`. Fashion loaders yield `(image, index)`, with the class label and the caption per index in `meta`.
- **Text encoder**: frozen CLIP ViT-B/32. TC-LoRA uses the pooled 512-vector; cross-attention uses the per-token embeddings with the padding mask. If CLIP cannot be loaded the notebook falls back to fixed random vectors and records `clip_fallback: true` in the manifest, so such a run is not mistaken for a result.
- **Backbone**: the Part A U-Net or the Part B MLP, unchanged, wrapped so the notebook can pass one condition id and `t` in [0, 1]. The embedding table has an extra null row, as in Part A.
- **Diffusion**: the cosine schedule the base was trained with, `add_noise` for training, and one sampler for every method: ancestral DDPM (`eta = 1`), 50 steps for images and 250 for cells, with the predicted clean state clamped to [-1, 1] for images and to each PC's training range for cells (`CFG_CLIP_X0=0` removes the clamp). TFG's sampler reduces to exactly this one when guidance is off, and the notebook asserts it.
- **Base model**: loads the `cfg_*` checkpoint (after checking that the split, schedule and backbone match), or a `base.pt` from an earlier run, or trains the base with the same recipe. Whichever happened, the manifest records it.
- **Method**: attach the TC-LoRA hypernetwork or the cross-attention block to the frozen base, assert that the adapted model equals the base at initialisation (both start at zero), and train only the adapter with the ordinary noise-prediction loss; or, for TFG, set up the classifier objective and assert that zero guidance reproduces plain sampling.
- **Diagnostics**: validation loss for the base with the null label, the base with the true label, and the adapter, on identical cells, timesteps and noise; the same pass with the wrong text (labels shifted by fixed offsets, so the noise is identical too); the loss gap by noise level; for TC-LoRA, the size of the generated correction relative to the frozen weights.
- **Samples**: every row (base, method, method with the wrong text or wrong target) starts from the same noise with the same generator seeds, on held-out conditions: for Tahoe the largest held-out pair of each of six drugs, for fashion a fixed set of classes. A scale check (variance of each row over that of the real data) runs before any metric and flags rows that left the data range.
- **Metrics**: Tahoe, per pair against its real validation cells: MMD, energy distance, Fréchet distance on the PCs, precision and recall, variance ratio, and drug accuracy under a k-nearest-neighbour vote over real training cells, with the same scorer's accuracy on the real cells of each pair as the ceiling. Fashion: FID (torchmetrics), top-1 and top-5 under a fine-tuned ResNet-18, and precision and recall in its feature space. The classifier that guides TFG is never the one that scores.
- **Manifest**: everything above, written to `manifest_<name>.json`.

### How the code works (`src/guidance/`)

- **`baselines/tc_lora.py`**: our reimplementation of TC-LoRA from the paper's Eq. 1 and Appendices A and B; no code was released. `HyperNet` is one network shared by all adapted layers: it takes a sinusoidal embedding of the timestep, the condition embedding and a learned embedding of the layer's depth and type, runs a residual trunk, and outputs the LoRA factors `A` and `B` for every layer at once. The head that writes `B` starts at zero, so the adapted model equals the base at the start. `AdaptedLinear` and `AdaptedConv2d` wrap a frozen layer and apply `W + (alpha / r) B A` with per-sample factors (for a convolution, as two 1x1 convolutions, which adapts the centre tap of the kernel). `TCLoRA` wraps a backbone, replaces its trunk layers with adapted ones, leaves the layers named in `skip_names` alone (timestep MLP, label embedding, output layer), and calls the hypernetwork once per forward pass. The header lists the seven places where this differs from the paper and why.
- **`baselines/cross_attention.py`**: PixArt-α's `MultiHeadCrossAttention` written in plain PyTorch. `TextCrossAttention` projects the feature map to queries and the text tokens to keys and values, masks padding, and adds the result back through an output projection that starts at zero, so a freshly attached block is the identity. `_Conditioned` wraps one stage of the backbone (run the stage, cross-attend, add). `TextConditioned` attaches one such block to a frozen backbone, at the bottleneck of the U-Net or after the input projection of the MLP, and exposes only the block's parameters for training.
- **`baselines/tfg.py`**: TFG's `guide_step` re-expressed against our sampler. `TFG.sample` does, per step, what Algorithm 1 of the paper does: predict the clean state and clamp it (a scalar range or per-dimension bounds); take the gradient of the smoothed objective through the denoiser (variance guidance); take `iter_steps` gradient steps on the clean estimate itself (mean guidance); apply their `rho`, `mu` and `sigma` schedules; optionally re-noise and repeat. `TFGStats` counts denoiser forward and backward passes and objective evaluations, which is how the notebooks report cost. With `rho = mu = 0` the sampler draws exactly the noise plain sampling draws, so the two agree bit for bit. Running the file executes its self-checks. The header lists what was kept from the official code and eight changes.
- **`data/cells.py`**: the `TahoePCs` loader from Part B, unchanged, plus `make_loaders`, which turns it into loaders that yield `(x, condition_id)`. The condition can be the drug, the (cell line, drug) pair, or the per-pair k-means cluster; `meta` carries the table from condition id to (drug id, line id), the text of every condition built from `panel_review.json`, and the DMSO reference per line. `find_pca_dir` looks for the PCA folder next to the notebook, under `data/` up to three folders up, on Colab and on Drive, or at `TAHOE_PCA_DIR`.
- **`data/fashion.py`**: the `FashionDataset` from the Modality-1 notebooks, unchanged (crop to the garment box, pad to a white square, resize, flip in training, pixels in [-1, 1], the official partition), plus `make_loaders`, which yields `(image, index)` and puts labels and captions in `meta`. `find_root` finds the DeepFashion folders or unpacks the two zips if only those are present.
- **`eval/cell_metrics.py`**: the metrics every Tahoe notebook uses: `mmd_rbf`, `energy_distance`, `frechet_distance` on the PCs, `precision_recall` with k-NN manifolds, `variance_ratio`, and `KNNConditionScorer`, a k-nearest-neighbour vote over real training cells. `all_metrics` returns them in one dict with the sample counts.
- **`eval/image_metrics.py`**: the evaluation pieces from the flow-matching Modality-1 notebook, shared by the four fashion notebooks: `FashionClassifier` (a ResNet on [-1, 1] images), `fit_classifier` (fine-tunes an ImageNet-pretrained ResNet on the real training images once and caches it, or loads `fashion_resnet18_classifier.pt` if present), `top_k_hits`, `real_accuracy`, and `precision_recall_features` in the classifier's feature space. FID comes from torchmetrics inside the notebooks.
