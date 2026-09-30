# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.5", "torchvision>=0.20", "numpy>=1.26", "Pillow>=10", "torchmetrics>=1.5", "torch-fidelity>=0.3"]
# ///
"""Class-conditional DeepFashion DDPM with classifier-free guidance.

Run with uv (from any directory):
  uv run --script fashion_cfg_diffusion.py check-data --data /path/to/fashion
  uv run --script fashion_cfg_diffusion.py train --data /path/to/fashion --out /path/to/run
  uv run --script fashion_cfg_diffusion.py sample --out /path/to/run --class-id 2
  uv run --script fashion_cfg_diffusion.py evaluate --data /path/to/fashion --out /path/to/run
  uv run --script fashion_cfg_diffusion.py train-classifier --data /path/to/fashion --out /path/to/run
  uv run --script fashion_cfg_diffusion.py evaluate-experiments --data /path/to/fashion --out /path/to/run

evaluate-experiments uses DATA/fashion_resnet18_classifier when that extracted
notebook checkpoint exists, or OUT/classifier_best.pt otherwise.

Class IDs printed by check-data and accepted by sample are zero-based. The
original DeepFashion category labels are one-based and converted on load.
"""

import argparse
import json
import math
import os
import random
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        next(handle)  # Count supplied by DeepFashion.
        next(handle)  # Column names.
        for line in handle:
            if line.strip():
                yield line.split()


class FashionData(Dataset):
    def __init__(self, root, split, size, flip=False, limit=0, seed=0):
        self.root = Path(root).expanduser().resolve()
        annotation = self.root / "Anno_coarse"
        self.classes = [" ".join(row[:-1]) for row in read_rows(annotation / "list_category_cloth.txt")]
        labels = {row[0]: int(row[1]) - 1 for row in read_rows(annotation / "list_category_img.txt")}
        boxes = {row[0]: tuple(map(int, row[1:5])) for row in read_rows(annotation / "list_bbox.txt")}
        records = []
        for path, partition in read_rows(self.root / "Eval/list_eval_partition.txt"):
            if partition == split and path in labels and path in boxes:
                label = labels[path]
                if not 0 <= label < len(self.classes):
                    raise ValueError(f"Invalid category label {label + 1} for {path}")
                records.append((path, label, boxes[path]))
        if not records:
            raise ValueError(f"No images for split {split!r} in {self.root}")
        if limit and limit < len(records):
            # Balanced, deterministic subset for limited-compute runs.
            by_class = defaultdict(list)
            for item in records:
                by_class[item[1]].append(item)
            rng = random.Random(seed)
            for group in by_class.values():
                rng.shuffle(group)
            keys = sorted(by_class)
            selected = []
            while len(selected) < limit and keys:
                for key in keys[:]:
                    if len(selected) == limit:
                        break
                    if by_class[key]:
                        selected.append(by_class[key].pop())
                    else:
                        keys.remove(key)
            records = selected
        self.records = records
        self.size = size
        self.flip = flip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative_path, label, box = self.records[index]
        path = self.root / relative_path
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = box
                x1, x2 = max(0, x1), min(image.width, x2)
                y1, y2 = max(0, y1), min(image.height, y2)
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
                image = ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BICUBIC,
                                     color=(255, 255, 255))
                if self.flip and random.random() < 0.5:
                    image = ImageOps.mirror(image)
                pixels = np.asarray(image, dtype=np.uint8).copy()
        except Exception as exc:
            raise RuntimeError(f"Cannot load {path}") from exc
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1)
        return tensor, label


def timestep_embedding(timesteps, width):
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(result, (0, width - result.shape[-1]))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_width):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_width, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.embedding(F.silu(embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.out(attention)


class UNet(nn.Module):
    def __init__(self, classes, base=64):
        super().__init__()
        if base % 8:
            raise ValueError("--base must be divisible by 8")
        width = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, width), nn.SiLU(), nn.Linear(width, width))
        self.class_embedding = nn.Embedding(classes + 1, width)
        self.input = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, width)
        self.stride1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.down2 = ResBlock(base, base * 2, width)
        self.stride2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.middle1 = ResBlock(base * 2, base * 2, width)
        self.attention = Attention(base * 2)
        self.middle2 = ResBlock(base * 2, base * 2, width)
        self.up2 = ResBlock(base * 4, base * 2, width)
        self.up1 = ResBlock(base * 3, base, width)
        self.output_norm = nn.GroupNorm(8, base)
        self.output = nn.Conv2d(base, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.base = base
        self.null_label = classes

    def forward(self, x, timesteps, labels):
        embedding = self.time_mlp(timestep_embedding(timesteps, self.base)) + self.class_embedding(labels)
        x = self.input(x)
        skip1 = self.down1(x, embedding)
        skip2 = self.down2(self.stride1(skip1), embedding)
        x = self.middle2(self.attention(self.middle1(self.stride2(skip2), embedding)), embedding)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(torch.cat([x, skip2], dim=1), embedding)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(torch.cat([x, skip1], dim=1), embedding)
        return self.output(F.silu(self.output_norm(x)))


class Diffusion:
    def __init__(self, device, count=1000):
        self.count = count
        steps = torch.linspace(0, count, count + 1, device=device, dtype=torch.float64) / count
        cumulative = torch.cos((steps + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0.0001, 0.9999).float()
        self.alpha_bar = torch.cumprod(1 - betas, dim=0)

    def noise(self, clean, noise, timesteps):
        alpha = self.alpha_bar[timesteps][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no accessible NVIDIA device is available")
    return torch.device(value)


def save_checkpoint(path, model, optimizer, scaler, epoch, step, config, classes,
                    best_val, significant_best, stale_epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "step": step,
                "config": config, "classes": classes, "best_val": best_val,
                "significant_best": significant_best,
                "stale_epochs": stale_epochs}, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def validate(model, diffusion, loader, device, seed):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    total = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        noise = torch.randn(images.shape, device=device, generator=rng)
        t = torch.randint(diffusion.count, (len(images),), device=device, generator=rng)
        corrupted = diffusion.noise(images, noise, t)
        prediction = model(corrupted, t, labels)
        total += F.mse_loss(prediction.float(), noise, reduction="sum").item()
        count += noise.numel()
    return total / count


def train(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    training = FashionData(args.data, "train", args.size, flip=True,
                           limit=args.max_train_images, seed=args.seed)
    validation = FashionData(args.data, "val", args.size,
                             limit=args.max_val_images, seed=args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    model = UNet(len(training.classes), args.base).to(device)
    diffusion = Diffusion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    active_classes = sorted({label for _, label, _ in training.records})
    config = {"size": args.size, "base": args.base, "diffusion_steps": diffusion.count,
              "data": str(Path(args.data).resolve()), "drop_probability": args.drop_probability,
              "seed": args.seed, "active_classes": active_classes}
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    start_epoch = step = 0
    best_val = significant_best = math.inf
    stale_epochs = 0
    if latest.exists():
        if not args.resume:
            raise FileExistsError(f"{latest} exists; pass --resume or select another --out")
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["config"] != config or state["classes"] != training.classes:
            raise ValueError("Checkpoint model/data settings differ from this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best_val = state["epoch"], state["step"], state["best_val"]
        # Older checkpoints predate early stopping; resume their best score
        # with a fresh patience counter.
        significant_best = state.get("significant_best", best_val)
        stale_epochs = state.get("stale_epochs", 0)
    (args.out / "classes.json").write_text(json.dumps(training.classes, indent=2) + "\n")
    print(f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, "
          f"train={len(training):,}, val={len(validation):,}, classes={len(training.classes)}, "
          f"starting_step={step}", flush=True)
    if device.type == "cpu":
        print("CPU training is substantially slower than CUDA training.", flush=True)
    if args.patience > 0 and stale_epochs >= args.patience:
        print(f"Early stopping was already reached after epoch {start_epoch}; "
              "increase --patience to continue this run.", flush=True)
        return
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = count = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = torch.where(torch.rand(labels.shape, device=device) < args.drop_probability,
                                 model.null_label, labels)
            noise = torch.randn_like(images)
            t = torch.randint(diffusion.count, (len(images),), device=device)
            corrupted = diffusion.noise(images, noise, t)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(corrupted, t, labels)
                loss = F.mse_loss(prediction.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running += loss.item() * len(images)
            count += len(images)
            if step % args.log_every == 0:
                print(f"epoch={epoch + 1} step={step} train_mse={running / count:.5f}", flush=True)
            if step % args.save_every == 0:
                save_checkpoint(latest, model, optimizer, scaler, epoch, step,
                                config, training.classes, best_val, significant_best,
                                stale_epochs)
            if args.max_steps and step >= args.max_steps:
                break
        score = validate(model, diffusion, val_loader, device, args.seed + 1000)
        if not math.isfinite(score):
            raise RuntimeError("Validation noise-prediction MSE is not finite")
        print(f"epoch={epoch + 1} step={step} train_mse={running / count:.5f} "
              f"val_mse={score:.5f}", flush=True)
        if score < significant_best - args.min_delta:
            significant_best, stale_epochs = score, 0
        else:
            stale_epochs += 1
        if score < best_val:
            best_val = score
            save_checkpoint(args.out / "best.pt", model, optimizer, scaler, epoch + 1,
                            step, config, training.classes, best_val, significant_best,
                            stale_epochs)
        save_checkpoint(latest, model, optimizer, scaler, epoch + 1, step,
                        config, training.classes, best_val, significant_best,
                        stale_epochs)
        should_stop = args.patience > 0 and stale_epochs >= args.patience
        with (args.out / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch + 1, "step": step,
                                     "train_mse": running / count, "val_mse": score,
                                     "best_val_mse": best_val,
                                     "epochs_without_significant_improvement": stale_epochs,
                                     "early_stopped": should_stop}) + "\n")
        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}: validation MSE did not "
                  f"improve by more than {args.min_delta:g} for {args.patience} "
                  f"epoch{'s' if args.patience != 1 else ''}. "
                  f"Best validation MSE={best_val:.5f}", flush=True)
            break
        if args.max_steps and step >= args.max_steps:
            break
    print(f"Saved: {latest} (best validation: {best_val:.5f})", flush=True)


def load_sampling_model(out, checkpoint_choice, device):
    checkpoint = out / ("best.pt" if checkpoint_choice == "best" else "latest.pt")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    classes, config = state["classes"], state["config"]
    model = UNet(len(classes), config["base"]).to(device).eval()
    model.load_state_dict(state["model"])
    diffusion = Diffusion(device, config["diffusion_steps"])
    return model, diffusion, classes, config, checkpoint


@torch.inference_mode()
def generate_images(model, diffusion, labels, size, sample_steps, guidance, rng):
    device = labels.device
    x = torch.randn((len(labels), 3, size, size), device=device, generator=rng)
    null = torch.full_like(labels, model.null_label)
    timesteps = torch.linspace(diffusion.count - 1, 0, sample_steps,
                               device=device).round().long().unique_consecutive()
    for index, t in enumerate(timesteps):
        time_batch = t.expand(len(labels))
        if guidance == 0:
            predicted = model(x, time_batch, null)
        elif guidance == 1:
            predicted = model(x, time_batch, labels)
        else:
            joined = model(torch.cat([x, x]), torch.cat([time_batch, time_batch]),
                           torch.cat([null, labels]))
            uncond, cond = joined.chunk(2)
            predicted = uncond + guidance * (cond - uncond)
        alpha = diffusion.alpha_bar[t]
        clean = ((x - (1 - alpha).sqrt() * predicted) / alpha.sqrt()).clamp(-1, 1)
        if index + 1 == len(timesteps):
            x = clean
        else:
            next_alpha = diffusion.alpha_bar[timesteps[index + 1]]
            x = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * predicted
    return x.clamp(-1, 1)


def validate_sample_options(args, classes, config):
    if not 0 <= args.class_id < len(classes):
        raise ValueError(f"--class-id must be between 0 and {len(classes) - 1}")
    if args.class_id not in config["active_classes"]:
        raise ValueError(f"Class {args.class_id} ({classes[args.class_id]}) has no training images")
    if not 1 <= args.sample_steps <= config["diffusion_steps"]:
        raise ValueError("--sample-steps is outside the diffusion schedule")


@torch.inference_mode()
def sample(args):
    device = choose_device(args.device)
    model, diffusion, classes, config, _ = load_sampling_model(args.out, args.checkpoint, device)
    validate_sample_options(args, classes, config)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    labels = torch.full((args.num_images,), args.class_id, device=device, dtype=torch.long)
    x = generate_images(model, diffusion, labels, config["size"], args.sample_steps,
                        args.guidance, rng)
    columns = math.ceil(math.sqrt(args.num_images))
    rows = math.ceil(args.num_images / columns)
    pixels = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().permute(0, 2, 3, 1).numpy()
    canvas = Image.new("RGB", (columns * config["size"], rows * config["size"]), "white")
    for index, pixels_one in enumerate(pixels):
        canvas.paste(Image.fromarray(pixels_one), ((index % columns) * config["size"],
                                                   (index // columns) * config["size"]))
    target = args.out / f"sample_class{args.class_id}_guidance{args.guidance:g}.png"
    canvas.save(target)
    print(f"Saved {target} — class {args.class_id}: {classes[args.class_id]}")


def to_fid_uint8(images):
    """TorchMetrics expects RGB uint8 NCHW images when normalize=False."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.inference_mode()
def evaluate(args):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError("FID requires torchmetrics and torch-fidelity; run with uv run --script") from exc
    device = choose_device(args.device)
    model, diffusion, classes, config, checkpoint = load_sampling_model(args.out, args.checkpoint, device)
    if not 1 <= args.sample_steps <= diffusion.count:
        raise ValueError("--sample-steps is outside the diffusion schedule")
    reference = FashionData(args.data, args.split, config["size"])
    if reference.classes != classes:
        raise ValueError("Reference category names do not match the checkpoint")
    indices = [index for index, (_, label, _) in enumerate(reference.records)
               if args.fid_scope == "overall" or label == args.class_id]
    if args.fid_scope == "class":
        validate_sample_options(args, classes, config)
    if len(indices) < 2:
        raise ValueError("FID requires at least two real reference images")
    selector = random.Random(args.seed)
    if args.num_real < len(indices):
        indices = selector.sample(indices, args.num_real)
    real_labels = [reference.records[index][1] for index in indices]
    generated_count = args.num_generated or len(indices)
    if generated_count < 2:
        raise ValueError("FID requires at least two generated images")
    if args.fid_scope == "overall":
        # Preserve the selected real set's category frequencies, including when
        # the requested generated count differs from the real count.
        generated_labels = (real_labels * math.ceil(generated_count / len(real_labels)))[:generated_count]
        selector.shuffle(generated_labels)
    else:
        generated_labels = [args.class_id] * generated_count
    try:
        metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("FID requires torch-fidelity; run with uv run --script") from exc
    metric.set_dtype(torch.float64)
    loader = DataLoader(torch.utils.data.Subset(reference, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"FID reference={args.split}/{args.fid_scope}, real={len(indices):,}, "
          f"generated={generated_count:,}, device={device}", flush=True)
    for batch_number, (images, _) in enumerate(loader, 1):
        metric.update(to_fid_uint8(images.to(device, non_blocking=True)), real=True)
        if batch_number % 25 == 0:
            print(f"real batches: {batch_number}/{len(loader)}", flush=True)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    batch_total = math.ceil(generated_count / args.batch_size)
    for batch_number, start in enumerate(range(0, generated_count, args.batch_size), 1):
        labels = torch.tensor(generated_labels[start:start + args.batch_size],
                              dtype=torch.long, device=device)
        images = generate_images(model, diffusion, labels, config["size"],
                                 args.sample_steps, args.guidance, rng)
        metric.update(to_fid_uint8(images), real=False)
        if batch_number % 25 == 0:
            print(f"generated batches: {batch_number}/{batch_total}", flush=True)
    score = float(metric.compute().item())
    if not math.isfinite(score):
        raise RuntimeError("FID returned a non-finite score")
    notes = []
    if min(len(indices), generated_count) <= 2048:
        notes.append("Small sample relative to 2048 Inception features; treat this score as a diagnostic.")
    report = {"metric": "FID", "score": score, "lower_is_better": True,
              "scope": args.fid_scope, "split": args.split,
              "class_id": args.class_id if args.fid_scope == "class" else None,
              "class_name": classes[args.class_id] if args.fid_scope == "class" else None,
              "real_images": len(indices), "generated_images": generated_count,
              "feature_dimension": 2048, "seed": args.seed,
              "guidance": args.guidance, "sample_steps": args.sample_steps,
              "checkpoint": str(checkpoint.resolve()), "data": str(reference.root),
              "batch_size": args.batch_size, "notes": notes}
    scope = f"class{args.class_id}" if args.fid_scope == "class" else "overall"
    target = args.report or args.out / f"fid_{scope}_{args.split}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FID={score:.4f}; saved report to {target}", flush=True)


class FashionClassifier(nn.Module):
    """ResNet-18 classifier with the notebook's ImageNet input preprocessing."""

    def __init__(self, classes, input_size=128, pretrained=False):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError("Classifier evaluation requires torchvision; run with uv run --script") from exc
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, classes)
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images, return_features=False):
        x = F.interpolate((images.clamp(-1, 1) + 1) / 2,
                          size=self.input_size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        features = net.avgpool(x).flatten(1)
        logits = net.fc(features)
        return (logits, features) if return_features else logits


def classifier_checkpoint_path(args):
    if args.classifier:
        return args.classifier
    trained = args.out / "classifier_best.pt"
    extracted = args.data / "fashion_resnet18_classifier"
    if args.command == "evaluate-experiments" and extracted.is_dir():
        return extracted
    return trained


def save_classifier(path, classifier, classes, size, epoch, val_top1, val_top5):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": classifier.state_dict(), "classes": classes, "size": size,
                "input_size": classifier.input_size, "epoch": epoch,
                "val_top1": val_top1, "val_top5": val_top5}, temporary)
    os.replace(temporary, path)


def load_classifier(path, classes, size, device):
    if not path.exists():
        raise FileNotFoundError(f"Classifier checkpoint {path} is missing")
    if path.is_dir():
        # A torch.save archive can be supplied as an extracted folder. Restore
        # its archive structure in a temporary file without changing the source.
        if not (path / "data.pkl").is_file() or not (path / "version").is_file():
            raise ValueError(f"{path} is not an extracted PyTorch checkpoint")
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as archive_file:
            with zipfile.ZipFile(archive_file, "w", compression=zipfile.ZIP_STORED,
                                 strict_timestamps=False) as archive:
                for item in path.rglob("*"):
                    if item.is_file():
                        archive.write(item, f"{path.name}/{item.relative_to(path)}")
            archive_file.seek(0)
            state = torch.load(archive_file, map_location="cpu", weights_only=True)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        if state["classes"] != classes or state["size"] != size:
            raise ValueError("Classifier categories or image size differ from the diffusion checkpoint")
        weights, input_size = state["model"], state["input_size"]
    elif isinstance(state, dict) and "backbone.fc.weight" in state:
        # The notebook saves a bare state_dict. Its FashionDataset reads the
        # same DeepFashion category file and converts labels to zero-based IDs.
        weights, input_size = state, 128
        if weights["backbone.fc.weight"].shape[0] != len(classes):
            raise ValueError("Classifier output count does not match the diffusion categories")
        state = {"model": weights, "input_size": input_size, "epoch": None,
                 "format": "notebook_state_dict"}
    else:
        raise ValueError(f"Unsupported classifier checkpoint format: {path}")
    classifier = FashionClassifier(len(classes), input_size).to(device)
    classifier.load_state_dict(weights)
    return classifier.eval(), state


def top_k_hits(logits, labels):
    ranked = logits.topk(min(5, logits.shape[1]), dim=1).indices
    matches = ranked.eq(labels[:, None])
    return matches[:, 0], matches.any(dim=1)


@torch.inference_mode()
def real_classifier_results(classifier, loader, device, collect_features=False):
    classifier.eval()
    top1 = top5 = count = 0
    features = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, embedding = classifier(images, return_features=True)
        hit1, hit5 = top_k_hits(logits, labels)
        top1 += int(hit1.sum())
        top5 += int(hit5.sum())
        count += len(labels)
        if collect_features:
            features.append(embedding.float().cpu())
    result = {"top1": top1 / count, "top5": top5 / count, "images": count}
    return result, torch.cat(features) if collect_features else None


def train_classifier(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    _, _, classes, config, diffusion_checkpoint = load_sampling_model(args.out, args.checkpoint, device)
    training = FashionData(args.data, "train", config["size"], flip=True,
                           limit=args.max_train_images, seed=args.seed)
    validation = FashionData(args.data, "val", config["size"],
                             limit=args.max_val_images, seed=args.seed)
    if training.classes != classes or validation.classes != classes:
        raise ValueError("Dataset categories do not match the diffusion checkpoint")
    path = classifier_checkpoint_path(args)
    if path.exists() and not args.overwrite_classifier:
        raise FileExistsError(f"{path} exists; pass --overwrite-classifier to retrain")
    training_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation, batch_size=args.batch_size,
                                   num_workers=args.workers, pin_memory=device.type == "cuda")
    classifier = FashionClassifier(len(classes), args.classifier_input_size,
                                   pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_top1 = -1.0
    print(f"Classifier train={len(training):,}, val={len(validation):,}, "
          f"classes={len(classes)}, device={device}, diffusion={diffusion_checkpoint}", flush=True)
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        loss_sum = count = 0
        for images, labels in training_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = classifier(images)
                loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(images)
            count += len(images)
        scheduler.step()
        validation_result, _ = real_classifier_results(classifier, validation_loader, device)
        print(f"Classifier epoch {epoch}: train_loss={loss_sum / count:.4f}, "
              f"real_val_top1={validation_result['top1']:.4f}, "
              f"real_val_top5={validation_result['top5']:.4f}", flush=True)
        if validation_result["top1"] > best_top1:
            best_top1 = validation_result["top1"]
            save_classifier(path, classifier, classes, config["size"], epoch,
                            validation_result["top1"], validation_result["top5"])
    print(f"Saved best classifier to {path} (validation top-1={best_top1:.4f})", flush=True)


def knn_radii(features, k, chunk_size=256):
    """Distance to the kth other point, excluding each point itself."""
    radii = []
    for start in range(0, len(features), chunk_size):
        distances = torch.cdist(features[start:start + chunk_size], features)
        rows = torch.arange(len(distances))
        distances[rows, start + rows] = float("inf")
        radii.append(distances.kthvalue(k, dim=1).values)
    return torch.cat(radii)


def feature_precision_recall(real_features, generated_features, k=3, chunk_size=256):
    """Improved precision/recall using separate real and generated k-NN manifolds."""
    real_features = real_features.float().cpu()
    generated_features = generated_features.float().cpu()
    if k < 1 or k >= min(len(real_features), len(generated_features)):
        raise ValueError("--pr-k must be smaller than both real and generated sample counts")
    real_radii = knn_radii(real_features, k, chunk_size)
    generated_radii = knn_radii(generated_features, k, chunk_size)
    precision_hits = recall_hits = 0
    for start in range(0, len(generated_features), chunk_size):
        distances = torch.cdist(generated_features[start:start + chunk_size], real_features)
        precision_hits += int((distances <= real_radii[None, :]).any(dim=1).sum())
    for start in range(0, len(real_features), chunk_size):
        distances = torch.cdist(real_features[start:start + chunk_size], generated_features)
        recall_hits += int((distances <= generated_radii[None, :]).any(dim=1).sum())
    return {"precision": precision_hits / len(generated_features),
            "recall": recall_hits / len(real_features)}


@torch.inference_mode()
def evaluate_experiments(args):
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    model, diffusion, classes, config, diffusion_checkpoint = load_sampling_model(
        args.out, args.checkpoint, device)
    if not 1 <= args.sample_steps <= diffusion.count:
        raise ValueError("--sample-steps is outside the diffusion schedule")
    classifier_path = classifier_checkpoint_path(args)
    classifier, classifier_state = load_classifier(classifier_path, classes, config["size"], device)
    reference = FashionData(args.data, args.split, config["size"])
    if reference.classes != classes:
        raise ValueError("Reference categories do not match the diffusion checkpoint")
    indices = list(range(len(reference)))
    if args.num_eval < len(indices):
        indices = random.Random(args.seed).sample(indices, args.num_eval)
    if len(indices) <= args.pr_k:
        raise ValueError("Evaluation requires more images than --pr-k")
    labels = [reference.records[index][1] for index in indices]
    loader = DataLoader(torch.utils.data.Subset(reference, indices), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    real_accuracy, real_features = real_classifier_results(classifier, loader, device,
                                                           collect_features=True)
    results = {}
    per_class_hits = {}
    for guidance in args.guidance_scales:
        rng = torch.Generator(device=device).manual_seed(args.seed)
        hit1_parts, hit5_parts, generated_features = [], [], []
        total_batches = math.ceil(len(labels) / args.batch_size)
        for batch_number, start in enumerate(range(0, len(labels), args.batch_size), 1):
            batch_labels = torch.tensor(labels[start:start + args.batch_size],
                                        dtype=torch.long, device=device)
            images = generate_images(model, diffusion, batch_labels, config["size"],
                                     args.sample_steps, guidance, rng)
            logits, features = classifier(images, return_features=True)
            hit1, hit5 = top_k_hits(logits, batch_labels)
            hit1_parts.append(hit1.cpu())
            hit5_parts.append(hit5.cpu())
            generated_features.append(features.float().cpu())
            if batch_number % 25 == 0:
                print(f"guidance={guidance:g}: generated {batch_number}/{total_batches} batches",
                      flush=True)
        hits1 = torch.cat(hit1_parts)
        hits5 = torch.cat(hit5_parts)
        pr = feature_precision_recall(real_features, torch.cat(generated_features), args.pr_k)
        results[f"{guidance:g}"] = {"top1": hits1.float().mean().item(),
                                    "top5": hits5.float().mean().item(), **pr}
        per_class_hits[guidance] = hits1
        print(f"guidance={guidance:g}: top1={results[f'{guidance:g}']['top1']:.4f}, "
              f"top5={results[f'{guidance:g}']['top5']:.4f}, "
              f"precision={pr['precision']:.4f}, recall={pr['recall']:.4f}", flush=True)
    highlighted_guidance = 3.0 if 3.0 in per_class_hits else max(per_class_hits)
    reference_labels = torch.tensor(labels)
    rows = []
    for class_id in sorted(set(labels)):
        mask = reference_labels == class_id
        if int(mask.sum()) >= 10:
            rows.append({"class_id": class_id, "class_name": classes[class_id],
                         "images": int(mask.sum()),
                         "top1": per_class_hits[highlighted_guidance][mask].float().mean().item()})
    rows.sort(key=lambda row: (row["top1"], row["class_id"]))
    report = {"experiment": "classifier_accuracy_and_precision_recall",
              "split": args.split, "real_images": len(indices),
              "generated_images_per_guidance": len(indices), "seed": args.seed,
              "sample_steps": args.sample_steps, "pr_k": args.pr_k,
              "real_accuracy": real_accuracy, "generated": results,
              "per_class_guidance": highlighted_guidance,
              "per_class_top1_min_10_images": rows,
              "diffusion_checkpoint": str(diffusion_checkpoint.resolve()),
              "classifier_checkpoint": str(classifier_path.resolve()),
              "classifier_epoch": classifier_state["epoch"],
              "classifier_format": classifier_state.get("format", "script_checkpoint"),
              "data": str(reference.root)}
    target = args.report or args.out / f"experiments_{args.split}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Real {args.split}: top1={real_accuracy['top1']:.4f}, "
          f"top5={real_accuracy['top5']:.4f}", flush=True)
    print(f"Guidance {highlighted_guidance:g} lowest classes: {rows[:5]}", flush=True)
    print(f"Guidance {highlighted_guidance:g} highest classes: {rows[-5:][::-1]}", flush=True)
    print(f"Saved experiment report to {target}", flush=True)


def check_data(args):
    for split in ("train", "val", "test"):
        data = FashionData(args.data, split, args.size)
        counts = Counter(label for _, label, _ in data.records)
        image, label = data[0]
        print(f"{split}: {len(data):,} images; {len(counts)} active classes; "
              f"first={data.classes[label]}; tensor={tuple(image.shape)}")
    print("Class IDs (zero-based):")
    for index, name in enumerate(data.classes):
        print(f"  {index:2d}  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check-data", "train", "sample", "evaluate",
                    "train-classifier", "evaluate-experiments"):
        sub = subparsers.add_parser(command)
        if command != "sample":
            sub.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
        if command in ("check-data", "train"):
            sub.add_argument("--size", type=int, default=64)
        if command != "check-data":
            sub.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "fashion-cfg-run")
            sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            sub.add_argument("--seed", type=int, default=42)
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--base", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--drop-probability", type=float, default=0.1)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--max-train-images", type=int, default=0)
    train_parser.add_argument("--max-val-images", type=int, default=2048)
    train_parser.add_argument("--max-steps", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument("--save-every", type=int, default=250)
    train_parser.add_argument("--patience", type=int, default=10,
                              help="Stop after this many epochs without a validation improvement; 0 disables")
    train_parser.add_argument("--min-delta", type=float, default=1e-4,
                              help="Minimum validation MSE decrease that resets patience")
    train_parser.add_argument("--resume", action="store_true")
    sample_parser = subparsers.choices["sample"]
    sample_parser.add_argument("--class-id", type=int, default=2)
    sample_parser.add_argument("--guidance", type=float, default=3.0)
    sample_parser.add_argument("--num-images", type=int, default=16)
    sample_parser.add_argument("--sample-steps", type=int, default=50)
    sample_parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    evaluate_parser = subparsers.choices["evaluate"]
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="test")
    evaluate_parser.add_argument("--fid-scope", choices=("overall", "class"), default="overall")
    evaluate_parser.add_argument("--class-id", type=int, default=2)
    evaluate_parser.add_argument("--num-real", type=int, default=2048)
    evaluate_parser.add_argument("--num-generated", type=int, default=None)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--workers", type=int, default=0)
    evaluate_parser.add_argument("--guidance", type=float, default=3.0)
    evaluate_parser.add_argument("--sample-steps", type=int, default=50)
    evaluate_parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    evaluate_parser.add_argument("--report", type=Path, default=None)
    classifier_parser = subparsers.choices["train-classifier"]
    classifier_parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    classifier_parser.add_argument("--classifier", type=Path, default=None,
                                   help="Classifier checkpoint path; default: OUT/classifier_best.pt")
    classifier_parser.add_argument("--classifier-input-size", type=int, default=128)
    classifier_parser.add_argument("--epochs", type=int, default=10)
    classifier_parser.add_argument("--lr", type=float, default=1e-3)
    classifier_parser.add_argument("--batch-size", type=int, default=16)
    classifier_parser.add_argument("--workers", type=int, default=0)
    classifier_parser.add_argument("--max-train-images", type=int, default=0)
    classifier_parser.add_argument("--max-val-images", type=int, default=2048)
    classifier_parser.add_argument("--no-pretrained", action="store_true",
                                   help="Initialize ResNet-18 randomly instead of using ImageNet weights")
    classifier_parser.add_argument("--overwrite-classifier", action="store_true")
    experiments_parser = subparsers.choices["evaluate-experiments"]
    experiments_parser.set_defaults(seed=123)
    experiments_parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    experiments_parser.add_argument("--classifier", type=Path, default=None,
                                    help="Classifier .pt file or extracted PyTorch checkpoint directory")
    experiments_parser.add_argument("--split", choices=("val", "test"), default="val")
    experiments_parser.add_argument("--num-eval", type=int, default=2048)
    experiments_parser.add_argument("--batch-size", type=int, default=16)
    experiments_parser.add_argument("--workers", type=int, default=0)
    experiments_parser.add_argument("--sample-steps", type=int, default=100)
    experiments_parser.add_argument("--guidance-scales", type=float, nargs="+", default=[0.0, 1.0, 3.0])
    experiments_parser.add_argument("--pr-k", type=int, default=3)
    experiments_parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if hasattr(args, "size") and (args.size < 16 or args.size % 4):
        parser.error("--size must be at least 16 and divisible by 4")
    if args.command == "train":
        if (args.epochs < 1 or args.batch_size < 1 or args.base < 8 or args.base % 8
                or args.workers < 0 or args.max_train_images < 0 or args.max_val_images < 0
                or args.max_steps < 0 or args.log_every < 1 or args.save_every < 1
                or args.patience < 0 or not math.isfinite(args.min_delta) or args.min_delta < 0
                or not 0 <= args.drop_probability < 1 or args.lr <= 0):
            parser.error("Invalid training option")
        train(args)
    elif args.command == "sample":
        if args.num_images < 1 or args.guidance < 0:
            parser.error("--num-images must be positive and --guidance nonnegative")
        sample(args)
    elif args.command == "evaluate":
        if (args.num_real < 2 or (args.num_generated is not None and args.num_generated < 2)
                or args.batch_size < 1 or args.workers < 0 or args.guidance < 0):
            parser.error("FID sample counts must be at least 2; batch size positive; guidance nonnegative")
        evaluate(args)
    elif args.command == "train-classifier":
        if (args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.workers < 0
                or args.max_train_images < 0 or args.max_val_images < 0
                or args.classifier_input_size < 32):
            parser.error("Invalid classifier training option")
        train_classifier(args)
    elif args.command == "evaluate-experiments":
        if (args.num_eval < 2 or args.batch_size < 1 or args.workers < 0
                or args.pr_k < 1 or not args.guidance_scales
                or any(not math.isfinite(g) or g < 0 for g in args.guidance_scales)):
            parser.error("Invalid experiment evaluation option")
        evaluate_experiments(args)
    else:
        check_data(args)


if __name__ == "__main__":
    main()
