"""Metrics for generated cell states, shared by every Tahoe notebook. All computed on the 50 PCs.

Quality and coverage of a generated population against the real cells of the same condition:
    mmd_rbf            maximum mean discrepancy with an RBF kernel, median-heuristic bandwidth
    energy_distance    2 E|x-y| - E|x-x'| - E|y-y'|; zero iff the distributions agree
    frechet_distance   the FID formula applied to the PCs directly (no Inception network):
                       ||mu_g - mu_r||^2 + Tr(S_g + S_r - 2 (S_g S_r)^{1/2}). Gaussian
                       assumption; needs well over 50 cells per side for the covariance
    precision_recall   Kynkaanniemi et al. 2019 (arXiv:1904.06991) with k-NN manifolds:
                       precision = share of generated cells inside the real manifold (quality),
                       recall = share of real cells inside the generated manifold (coverage)
    variance_ratio     total variance of the samples over that of the real cells; far below 1
                       is collapse

Condition adherence without a trained classifier:
    KNNConditionScorer a k-nearest-neighbour vote over real training cells labelled with their
                       drug (or any label). Fit once, apply to generated cells. Its accuracy on
                       real validation cells is the ceiling and should be reported next to any
                       generated number.

Everything takes numpy arrays or torch tensors of shape (n, d) and returns Python floats.
"""

import numpy as np
import torch


def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def _sub(a, max_n, rng):
    return a[rng.choice(len(a), max_n, replace=False)] if len(a) > max_n else a


def mmd_rbf(x, y, bandwidth=None, max_n=500, seed=0):
    """Returns (mmd^2, bandwidth). Median heuristic for the bandwidth unless given."""
    rng = np.random.default_rng(seed)
    x, y = _sub(_np(x), max_n, rng), _sub(_np(y), max_n, rng)
    z = np.vstack([x, y]).astype(np.float64)
    d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    if bandwidth is None:
        med = np.median(d2[d2 > 0])
        bandwidth = float(np.sqrt(med / 2)) if med > 0 else 1.0
    k = np.exp(-d2 / (2 * bandwidth ** 2))
    n, m = len(x), len(y)
    kxx = (k[:n, :n].sum() - np.trace(k[:n, :n])) / max(n * (n - 1), 1)
    kyy = (k[n:, n:].sum() - np.trace(k[n:, n:])) / max(m * (m - 1), 1)
    return float(kxx + kyy - 2 * k[:n, n:].mean()), bandwidth


def energy_distance(x, y, max_n=500, seed=0):
    rng = np.random.default_rng(seed)
    x, y = _sub(_np(x), max_n, rng).astype(np.float64), _sub(_np(y), max_n, rng).astype(np.float64)
    d = lambda a, b: np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).mean()
    return float(2 * d(x, y) - d(x, x) - d(y, y))


def variance_ratio(pred, real):
    pred, real = _np(pred), _np(real)
    return float(np.trace(np.cov(pred, rowvar=False)) / np.trace(np.cov(real, rowvar=False)))


def frechet_distance(pred, real, eps=1e-6):
    """FID formula on the vectors themselves. Tr((S_g S_r)^{1/2}) via the eigenvalues of
    S_r^{1/2} S_g S_r^{1/2}, which is symmetric, so no complex square roots."""
    pred, real = _np(pred).astype(np.float64), _np(real).astype(np.float64)
    mu_g, mu_r = pred.mean(0), real.mean(0)
    s_g, s_r = np.cov(pred, rowvar=False), np.cov(real, rowvar=False)
    w, v = np.linalg.eigh(s_r)
    r_half = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = r_half @ s_g @ r_half
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu_g - mu_r) ** 2).sum() + np.trace(s_g) + np.trace(s_r) - 2 * tr_sqrt)


def precision_recall(real, pred, k=5, max_n=1000, seed=0):
    """Improved precision and recall (Kynkaanniemi et al. 2019) with k-NN radii.
    precision: fraction of generated points within some real point's k-NN ball.
    recall:    fraction of real points within some generated point's k-NN ball."""
    rng = np.random.default_rng(seed)
    real, pred = _sub(_np(real), max_n, rng).astype(np.float64), _sub(_np(pred), max_n, rng).astype(np.float64)

    def radii(a):
        d = np.sqrt(((a[:, None] - a[None]) ** 2).sum(-1))
        return np.sort(d, axis=1)[:, min(k, len(a) - 1)]

    def covered(points, manifold, r):
        d = np.sqrt(((points[:, None] - manifold[None]) ** 2).sum(-1))       # (n_points, n_manifold)
        return float((d <= r[None]).any(1).mean())

    return covered(pred, real, radii(real)), covered(real, pred, radii(pred))


def all_metrics(pred, real, seed=0):
    """Every distribution metric in one dict, with the counts that qualify them."""
    m, bw = mmd_rbf(pred, real, seed=seed)
    p, r = precision_recall(real, pred, seed=seed)
    return {"mmd": m, "mmd_bandwidth": bw, "energy": energy_distance(pred, real, seed=seed),
            "frechet_pc": frechet_distance(pred, real), "precision": p, "recall": r,
            "var_ratio": variance_ratio(pred, real), "n_generated": int(len(pred)), "n_real": int(len(real))}


class KNNConditionScorer:
    """k-nearest-neighbour vote over real reference cells with their labels."""

    def __init__(self, x_ref, y_ref, k=25, max_ref=100_000, seed=0, device=None):
        x_ref, y_ref = torch.as_tensor(x_ref), torch.as_tensor(y_ref)
        if len(x_ref) > max_ref:
            idx = torch.randperm(len(x_ref), generator=torch.Generator().manual_seed(seed))[:max_ref]
            x_ref, y_ref = x_ref[idx], y_ref[idx]
        self.device = device or x_ref.device
        self.x, self.y, self.k = x_ref.to(self.device).float(), y_ref.to(self.device).long(), k

    @torch.no_grad()
    def predict(self, x, chunk=1024):
        x = torch.as_tensor(x).to(self.device).float()
        out = []
        for i in range(0, len(x), chunk):
            d = torch.cdist(x[i:i + chunk], self.x)
            out.append(torch.mode(self.y[d.topk(self.k, largest=False).indices], dim=1).values)
        return torch.cat(out)

    def accuracy(self, x, y):
        y = torch.as_tensor(y).to(self.device).long()
        return float((self.predict(x) == y).float().mean())

    def describe(self):
        return f"{self.k}-nearest-neighbour vote over {len(self.x):,} real reference cells"


if __name__ == "__main__":
    g = np.random.default_rng(0)
    real = g.normal(size=(600, 50))
    same = g.normal(size=(600, 50))
    shifted = g.normal(size=(600, 50)) + 1.0
    narrow = g.normal(size=(600, 50)) * 0.3
    for name, pred in [("same distribution", same), ("shifted", shifted), ("collapsed", narrow)]:
        r = all_metrics(pred, real)
        print(f"{name:18s} mmd {r['mmd']:.3f}  energy {r['energy']:.3f}  frechet {r['frechet_pc']:.2f}  "
              f"precision {r['precision']:.2f}  recall {r['recall']:.2f}  var {r['var_ratio']:.2f}")
    base = all_metrics(same, real); sh = all_metrics(shifted, real); nr = all_metrics(narrow, real)
    assert sh["mmd"] > base["mmd"] and sh["frechet_pc"] > base["frechet_pc"] and sh["energy"] > base["energy"]
    assert nr["var_ratio"] < 0.2 and nr["recall"] < base["recall"]
    x = torch.tensor(np.vstack([real, real + 3]), dtype=torch.float32); y = torch.tensor([0] * 600 + [1] * 600)
    sc = KNNConditionScorer(x, y, k=5)
    assert sc.accuracy(x[:600] + 0.1, torch.zeros(600, dtype=torch.long)) > 0.95
    print("ok:", sc.describe())
