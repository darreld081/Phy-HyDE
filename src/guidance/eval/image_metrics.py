"""Evaluation pieces for the DeepFashion image notebooks, shared by all four of them.

`FashionClassifier`, `fit_classifier`, `top_k_hits`, `knn_radii` and `precision_recall_features`
are from a teammate's CIS6720_Proj1_FlowMatchingCode_Modality_1.ipynb (Experiment 3 and the
precision/recall cell), kept as written apart from being parameterised by architecture and
taking loaders that yield (image, index). No pretrained DeepFashion classifier exists, so two
ImageNet-pretrained ResNets are fine-tuned on the real training images, once, and cached:
one guides (TFG's objective), the other scores (accuracy, precision, recall), so guidance is
never judged by its own objective. Both are trained on real images only, never on samples.

FID itself comes from torchmetrics (Inception pool features, 2048-d) in the notebooks.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLF_INPUT_SIZE = 128     # 64x64 images are upsampled inside the wrapper to suit the ImageNet-pretrained backbone.


class FashionClassifier(nn.Module):
    """ResNet that takes images in [-1, 1] (the same range as the generative model) and returns class logits."""

    def __init__(self, num_classes, arch="resnet18"):
        super().__init__()
        ctor = getattr(models, arch)
        weights = getattr(models, f"{arch.replace('resnet', 'ResNet')}_Weights").IMAGENET1K_V1
        self.backbone = ctor(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.arch = arch

    def _prep(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=CLF_INPUT_SIZE, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.backbone(self._prep(x))

    @torch.no_grad()
    def features(self, x):
        """Penultimate features (ResNet without its fc layer), for precision / recall."""
        body = nn.Sequential(*list(self.backbone.children())[:-1])
        return body(self._prep(x)).flatten(1).float()


def top_k_hits(logits, labels, ks=(1, 5)):
    """Boolean [B] tensor per k: is the true label among the k highest-scoring classes?"""
    top = logits.topk(max(ks), dim=1).indices
    hits = top == labels[:, None]
    return [hits[:, :k].any(dim=1) for k in ks]


@torch.no_grad()
def real_accuracy(classifier, loader, labels_of, device):
    """Top-1 / top-5 on a loader yielding (image, index); `labels_of` maps index -> class."""
    classifier.eval()
    top1 = top5 = total = 0
    for images, idx in loader:
        images, y = images.to(device), labels_of[idx].to(device)
        hit1, hit5 = top_k_hits(classifier(images), y)
        top1, top5, total = top1 + hit1.sum().item(), top5 + hit5.sum().item(), total + len(images)
    return top1 / total, top5 / total


def fit_classifier(arch, num_classes, train_loader, val_loader, labels_train, labels_val, device, path,
                   epochs=10, lr=1e-3, seed=0, log=print, search=()):
    """Fine-tune an ImageNet-pretrained ResNet on the real training images; cache to `path`.
    Their recipe: AdamW 1e-3 with weight decay 1e-4, cosine annealing, mixed precision, `epochs` epochs.
    `search`: further paths tried before training, e.g. the checkpoint the flow-matching notebook
    saved (`fashion_resnet18_classifier.pt`), so one fine-tuned scorer serves every notebook."""
    path = Path(path)
    torch.manual_seed(seed)
    clf = FashionClassifier(num_classes, arch).to(device)
    found = next((Path(p) for p in [path, *search] if p and Path(p).is_file()), None)
    if found is not None:
        clf.load_state_dict(torch.load(found, map_location=device, weights_only=True))
        log(f"loaded fine-tuned {arch} from {found}")
        return clf.eval()
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    for epoch in range(epochs):
        clf.train()
        total_loss = num_images = 0
        for images, idx in train_loader:
            images, y = images.to(device), labels_train[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = F.cross_entropy(clf(images).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * len(images)
            num_images += len(images)
        sched.step()
        top1, top5 = real_accuracy(clf, val_loader, labels_val, device)
        log(f"{arch} epoch {epoch + 1}: train loss={total_loss / num_images:.4f}, "
            f"real validation top-1={top1:.3f}, top-5={top5:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf.eval()


def knn_radii(features, k):
    distances = torch.cdist(features, features)
    return distances.kthvalue(k + 1, dim=1).values  # k+1 because each point is its own nearest neighbour.


def precision_recall_features(real_features, generated_features, k=3):
    """Precision = share of generated points inside the real k-NN manifold (fidelity).
    Recall = share of real points inside the generated k-NN manifold (diversity / coverage)."""
    real_radii = knn_radii(real_features, k)
    generated_radii = knn_radii(generated_features, k)
    distances = torch.cdist(real_features, generated_features)  # [num_real, num_generated]
    precision = (distances <= real_radii[:, None]).any(dim=0).float().mean().item()
    recall = (distances <= generated_radii[None, :]).any(dim=1).float().mean().item()
    return precision, recall


def to_uint8(images):
    """FID's Inception implementation expects RGB uint8 pixels in [0, 255]."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    real = torch.randn(300, 64, generator=g)
    same, shifted = torch.randn(300, 64, generator=g), torch.randn(300, 64, generator=g) + 3
    p1, r1 = precision_recall_features(real, same); p2, r2 = precision_recall_features(real, shifted)
    print(f"same: precision {p1:.2f} recall {r1:.2f} | shifted: precision {p2:.2f} recall {r2:.2f}")
    assert p1 > 0.5 and r1 > 0.5 and p2 < 0.1 and r2 < 0.1
    clf = FashionClassifier(6, "resnet18").eval()
    x = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    print("logits", tuple(clf(x).shape), "| features", tuple(clf.features(x).shape))
    print("ok")
