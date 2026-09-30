"""Training-free guidance, following TFG.

PAPER
    Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma,
    James Zou, Stefano Ermon.
    "TFG: Unified Training-Free Guidance for Diffusion Models." NeurIPS 2024.
    arXiv:2409.15761.

CODE PROVENANCE
    Official code: https://github.com/YWolfeee/Training-Free-Guidance (MIT), read
    2026-09-28. The pieces that matter are `TFGGuidance.guide_step` and
    `tilde_get_guidance` in methods/tfg.py, the `_predict_*` helpers in methods/base.py,
    and `ImageLabelGuidance` in tasks/image_label_guidance.py. We did not import their
    package: it is built around diffusers pipelines and their own task/network registry.
    `TFG.sample` below is a re-expression of `guide_step` against our DDIM sampler, kept
    line-for-line close so it can be audited against theirs.

WHAT WE FOLLOW FROM THEIR CODE (Algorithm 1 of the paper)
    Per timestep, for each of N_recur recurrence steps:
      1. x_{0|t} = (x_t - sqrt(1 - abar_t) eps(x_t, t)) / sqrt(abar_t), clamped.
      2. Variance guidance  Delta_t = rho_t * grad_{x_t} log f~(x_{0|t}),
         the gradient taken THROUGH the denoiser.
      3. Mean guidance      Delta_0 accumulated over N_iter steps of
                            Delta_0 += mu_t * grad_{x_0} log f~(x_{0|t} + Delta_0).
      4. x_{t-1} = DDIM(x_t, x_{0|t}) + Delta_t / sqrt(alpha_t) + Delta_0 * sqrt(abar_{t-1}).
         The DDIM step here is their `_predict_x_prev_from_zero`: it RECOMPUTES eps from
         the clipped x_{0|t} before stepping. A sampler that clips x0 but keeps the
         original eps is a different update whenever the clamp engages, and the two can
         differ by O(1) on real images. Any "plain DDIM" this is compared against must
         use the same convention, or the rho = mu = 0 equality check fails.
      5. If recurring, re-noise x_{t-1} back to level t.
    f~ is the smoothed objective: log f~(x) = logsumexp_i log f(x + sigma_t * delta_i)
    minus log(eps_bsz), delta_i ~ N(0, I). The schedules for rho_t, mu_t and sigma_t are
    theirs (`get_rho`, `get_mu`, `get_std`), including the normalisation that makes the
    mean of rho_t over the trajectory equal to the `rho` you pass.

    The defaults below are their CIFAR-10 label-guidance script (scripts/cifar10_label.sh):
    rho=1, mu=0.25, sigma=0.001, eps_bsz=1, iter_steps=4, recur_steps=1, 50 steps,
    clip_x0=True, schedules increase/increase/decrease. Note that with eps_bsz=1 and
    sigma=0.001 the smoothing is effectively off in that configuration.

WHAT WE CHANGED, AND WHY
    1. Sampler stochasticity. Their CIFAR-10 script uses eta=1.0 (stochastic DDIM). Our
       other arms sample with deterministic DDIM (eta=0), so the default here is eta=0 to
       keep the comparison matched. The eta term is implemented and can be set to 1.0 to
       reproduce their setting.
    2. Delta_t scaling follows the released code, Delta_t / sqrt(alpha_t). (An earlier
       version of this header claimed the paper writes alpha_t; that came from a text
       extraction that dropped radical signs and should not be relied on.)
    3. Objective interface. Their `ImageLabelGuidance` wraps a ResNet that already returns
       the target class's log-probability. We take a plain callable `log_p_fn(x0) -> (B,)`
       so the same sampler serves a pretrained image classifier (with its own input
       normalisation) and a soft k-means objective on cell states.
    4. Base model. Their diffusion model is unconditional. Ours is class-conditional
       with a null label, so we run it with the null label to obtain the unconditional
       denoiser TFG expects. This is stated in the notebook and the manifest.
    5. `rescale_grad` is dropped: in their code it only acts when a molecule `node_mask`
       is supplied and is the identity for images.
    6. Cost accounting. Each guided step costs one denoiser forward plus one backward
       through it (for Delta_t) plus N_iter objective forward/backward passes. We count
       denoiser and objective evaluations separately and report both, because our own
       method's pitch is a single cheap forward pass.
    7. The clamp can be per dimension. Their `clip_sample` is the scalar [-1, 1] of images.
       `clip_x0` also accepts a tensor of per-dimension bounds (the training range of
       standardised cell states), or a (lo, hi) pair of tensors. This matters more than it
       looks: x_{0|t} = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t) divides by sqrt(abar_t),
       which is ~1e-4 at the first step of a cosine schedule (2e-9 under diffusers'
       squaredcos_cap_v2), so any error in eps, and any error in its Jacobian that Delta_t
       goes through, is amplified by 1e4 or more there. On images the [-1, 1] clamp
       saturates and zeroes that gradient; with no clamp at all (as we first ran on cells)
       the guided samples left the data range by a factor of 300 in standard deviation.
    8. Noise bookkeeping. With rho = mu = 0 the sampler draws exactly the noise plain
       DDIM/DDPM draws from `generator`, so a guidance-off run is bit-identical to the
       unguided sampler at any eta: the smoothing noise is only drawn when an objective
       is evaluated, and the recurrence re-noising is not drawn on the last recurrence
       pass (their loop draws it and discards it). Neither changes the guided update.

Run this file directly to execute the self-checks at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class TFGConfig:
    """Defaults are the official CIFAR-10 label-guidance script."""
    rho: float = 1.0
    mu: float = 0.25
    sigma: float = 0.001
    eps_bsz: int = 1
    iter_steps: int = 4
    recur_steps: int = 1
    rho_schedule: str = "increase"       # increase | decrease | constant
    mu_schedule: str = "increase"
    sigma_schedule: str = "decrease"     # decrease | constant
    # Clamp of the predicted x0. A float c clamps to [-c, c] (0 or None disables); a tensor b
    # of shape (*data_shape) clamps to [-b, b] per dimension; a (lo, hi) pair of tensors clamps
    # to [lo, hi] per dimension. Deviation 7 in the header.
    clip_x0: object = 1.0
    eta: float = 0.0                     # 0 = deterministic DDIM; their script used 1.0


@dataclass
class TFGStats:
    denoiser_evals: int = 0              # forward passes of eps_fn
    denoiser_backward: int = 0           # backward passes through eps_fn (Delta_t)
    objective_evals: int = 0             # forward+backward passes of log_p_fn
    steps: int = 0


class TFG:
    """Training-free guidance around a DDIM sampler.

    eps_fn(x, t) -> predicted noise, t a float tensor in [0, 1] (our convention).
    log_p_fn(x0) -> (B,) log-probability of the target under a predictor on CLEAN data.
    abar: (K,) cumulative alpha schedule the denoiser was trained with.
    """

    def __init__(self, eps_fn, log_p_fn, abar: torch.Tensor, cfg: TFGConfig = None):
        self.eps_fn, self.log_p_fn = eps_fn, log_p_fn
        self.abar = abar
        self.cfg = cfg or TFGConfig()
        self.K = abar.shape[0]
        self.stats = TFGStats()
        self._bounds = self.clamp_bounds(self.cfg.clip_x0, device=abar.device)

    # -- schedules, verbatim from methods/tfg.py ------------------------------------------
    @staticmethod
    def _schedule(kind, ap, ap_prev):
        if kind == "decrease":            # beta_t
            return 1 - ap / ap_prev
        if kind == "increase":            # alpha_t
            return ap / ap_prev
        if kind == "constant":
            return torch.ones_like(ap)
        raise ValueError(kind)

    def _strength(self, base, kind, i, ap, ap_prev):
        s = self._schedule(kind, ap, ap_prev)
        return base * s[i] * len(s) / s.sum()          # mean over the trajectory == base

    def _std(self, i, ap):
        if self.cfg.sigma_schedule == "decrease":
            return self.cfg.sigma * (1 - ap[i]).sqrt()
        return torch.tensor(self.cfg.sigma, device=ap.device)

    # -- objective with smoothing, verbatim logic of tilde_get_guidance ------------------
    def _log_f_tilde(self, x0, mc_eps):
        """logsumexp over eps_bsz noisy copies, minus log(eps_bsz). Returns (B,)."""
        flat = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        lp = self.log_p_fn(flat).reshape(mc_eps.shape[0], x0.shape[0])
        self.stats.objective_evals += 1
        return torch.logsumexp(lp, dim=0) - math.log(mc_eps.shape[0])

    @staticmethod
    def clamp_bounds(clip_x0, device=None):
        """Normalise `clip_x0` to (lo, hi) tensors, or None when clamping is off.
        Shared with the notebooks' plain sampler so both sides clamp identically."""
        if clip_x0 is None:
            return None
        if isinstance(clip_x0, (tuple, list)):
            lo, hi = (torch.as_tensor(b, dtype=torch.float32, device=device) for b in clip_x0)
            return lo, hi
        if isinstance(clip_x0, torch.Tensor):
            hi = clip_x0.to(device=device or clip_x0.device, dtype=torch.float32).abs()
            return -hi, hi
        if float(clip_x0) == 0.0:
            return None
        c = float(clip_x0)
        return torch.tensor(-c, device=device), torch.tensor(c, device=device)

    @staticmethod
    def clamp_x0(x0, bounds):
        """Clamp x0 to (lo, hi); bounds broadcast over the batch. None leaves x0 as is."""
        if bounds is None:
            return x0
        lo, hi = bounds
        return torch.maximum(torch.minimum(x0, hi.to(x0.device)), lo.to(x0.device))

    def _predict_x0(self, x, eps, ap):
        x0 = (x - (1 - ap).sqrt() * eps) / ap.sqrt()
        return self.clamp_x0(x0, self._bounds)

    @torch.no_grad()
    def sample(self, n, shape, steps, x_init=None, generator=None, n_snapshots=0):
        """Guided DDIM from noise. Mirrors TFGGuidance.guide_step per timestep."""
        c = self.cfg
        dev = self.abar.device
        self.stats = TFGStats(steps=steps)

        idx = torch.linspace(self.K - 1, 0, steps).long().to(dev)
        ap_all = self.abar[idx]                                          # alpha_prod_ts
        ap_prev_all = torch.cat([self.abar[idx[1:]], torch.ones(1, device=dev)])

        x = torch.randn(n, *shape, device=dev, generator=generator) if x_init is None \
            else x_init.clone()
        keep = set(torch.linspace(0, steps - 1, max(n_snapshots, 1)).long().tolist())
        snaps = [x.cpu().clone()] if n_snapshots else []

        for i in range(steps):
            ap, ap_prev = ap_all[i], ap_prev_all[i]
            t = (idx[i].float() / self.K).repeat(n)
            rho = self._strength(c.rho, c.rho_schedule, i, ap_all, ap_prev_all)
            mu = self._strength(c.mu, c.mu_schedule, i, ap_all, ap_prev_all)
            std = self._std(i, ap_all)
            guided = float(rho) != 0.0 or (float(mu) != 0.0 and c.iter_steps > 0)

            for r in range(c.recur_steps):
                # Smoothing noise for log f~, drawn only when an objective will be evaluated,
                # so that rho = mu = 0 consumes exactly the noise plain DDIM/DDPM consumes.
                mc_eps = (torch.zeros((1, *x.shape), device=dev) if not guided or float(std) == 0.0 else
                          torch.randn((c.eps_bsz, *x.shape), device=dev, generator=generator) * std)

                # Variance guidance: gradient of the smoothed objective w.r.t. x_t,
                # through the denoiser.
                if float(rho) != 0.0:
                    with torch.enable_grad():
                        x_g = x.detach().requires_grad_(True)
                        x0 = self._predict_x0(x_g, self.eps_fn(x_g, t), ap)
                        lp = self._log_f_tilde(x0, mc_eps)
                        delta_t = torch.autograd.grad(lp.sum(), x_g)[0] * rho
                    self.stats.denoiser_evals += 1
                    self.stats.denoiser_backward += 1
                    x0 = x0.detach()
                else:
                    delta_t = torch.zeros_like(x)
                    x0 = self._predict_x0(x, self.eps_fn(x, t), ap)
                    self.stats.denoiser_evals += 1

                # Mean guidance: N_iter gradient steps on the clean estimate itself.
                new_x0 = x0.clone()
                for _ in range(c.iter_steps):
                    if float(mu) != 0.0:
                        with torch.enable_grad():
                            x0_g = new_x0.detach().requires_grad_(True)
                            lp = self._log_f_tilde(x0_g, mc_eps)
                            g = torch.autograd.grad(lp.sum(), x0_g)[0]
                        new_x0 = new_x0 + mu * g
                delta_0 = new_x0 - x0

                # DDIM step from (x_t, x_{0|t}): recompute eps from x0, DDIM eq. 12.
                eps_new = (x - ap.sqrt() * x0) / (1 - ap).sqrt()
                sigma_ddim = c.eta * ((1 - ap_prev) / (1 - ap) * (1 - ap / ap_prev)).sqrt()
                x_prev = (ap_prev.sqrt() * x0
                          + (1 - ap_prev - sigma_ddim ** 2).clamp(min=0).sqrt() * eps_new)
                if c.eta > 0:
                    x_prev = x_prev + sigma_ddim * torch.randn(x.shape, device=dev, generator=generator)

                alpha_t = ap / ap_prev
                x_prev = x_prev + delta_t / alpha_t.sqrt() + delta_0 * ap_prev.sqrt()

                # Recurrence: re-noise x_{t-1} to level t and go round again. Not on the last
                # pass: that draw would be discarded and would only desynchronise `generator`
                # from the plain sampler.
                if r + 1 < c.recur_steps:
                    x = (alpha_t.sqrt() * x_prev
                         + (1 - alpha_t).sqrt() * torch.randn(x.shape, device=dev, generator=generator))

            x = x_prev.detach()
            if n_snapshots and i in keep:
                snaps.append(x.cpu().clone())

        return (x, snaps) if n_snapshots else x


if __name__ == "__main__":
    torch.manual_seed(0)
    K, D, n = 1000, 8, 64

    # Cosine schedule, as in the notebooks.
    s_ = torch.arange(K + 1, dtype=torch.float32) / K
    f_ = torch.cos((s_ + 0.008) / 1.008 * math.pi / 2) ** 2
    abar = (f_ / f_[0])[1:].clamp(1e-5, 0.9999)

    # The exact denoiser for data ~ N(0, I): E[x0 | x_t] = sqrt(abar) x_t, so
    # eps = sqrt(1 - abar) x_t. Its x0 estimate depends on x_t, which variance guidance
    # needs (a denoiser with zero Jacobian would make Delta_t identically zero).
    def eps_fn(x, t):
        a = abar[(t * K).long().clamp(0, K - 1)].view(-1, *([1] * (x.dim() - 1)))
        return (1 - a).sqrt() * x

    target = torch.full((D,), 2.0)
    def log_p_fn(x0):                    # log N(x0; target, I), up to a constant
        return -0.5 * ((x0 - target) ** 2).sum(-1)

    x_init = torch.randn(n, D)

    # 1. rho = mu = 0 reproduces plain deterministic DDIM exactly.
    def plain_ddim(x, steps):
        idx = torch.linspace(K - 1, 0, steps).long()
        for i, k in enumerate(idx):
            a = abar[k]; a_prev = abar[idx[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            t = (k.float() / K).repeat(x.shape[0])
            eps = eps_fn(x, t)
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x

    off = TFG(eps_fn, log_p_fn, abar, TFGConfig(rho=0, mu=0, iter_steps=0, clip_x0=0))
    plain = off.sample(n, (D,), 20, x_init=x_init)
    assert torch.allclose(plain, plain_ddim(x_init, 20), atol=1e-5)
    assert off.stats.denoiser_backward == 0 and off.stats.objective_evals == 0
    print("rho = mu = 0 reproduces plain DDIM: no objective calls, no backward through the denoiser")

    # 2. Guidance moves samples toward the target.
    on = TFG(eps_fn, log_p_fn, abar, TFGConfig(clip_x0=0))
    guided = on.sample(n, (D,), 20, x_init=x_init)
    d_plain = (plain - target).norm(dim=-1).mean().item()
    d_guided = (guided - target).norm(dim=-1).mean().item()
    assert d_guided < d_plain, (d_plain, d_guided)
    print(f"distance to target: plain {d_plain:.3f} -> guided {d_guided:.3f}")

    # 3. Cost accounting matches the algorithm.
    st = on.stats
    assert st.denoiser_evals == 20 and st.denoiser_backward == 20
    assert st.objective_evals == 20 * (1 + on.cfg.iter_steps)      # Delta_t + N_iter of Delta_0
    print(f"per sample: {st.denoiser_evals} denoiser fwd, {st.denoiser_backward} denoiser bwd, "
          f"{st.objective_evals} objective fwd+bwd  (plain DDIM would be 20 fwd)")

    # 4. Mean guidance alone (rho=0) and variance guidance alone (mu=0) both help.
    mean_only = TFG(eps_fn, log_p_fn, abar, TFGConfig(rho=0, clip_x0=0)).sample(n, (D,), 20, x_init=x_init)
    var_only = TFG(eps_fn, log_p_fn, abar, TFGConfig(mu=0, iter_steps=0, clip_x0=0)).sample(n, (D,), 20, x_init=x_init)
    for name, s in [("mean guidance only", mean_only), ("variance guidance only", var_only)]:
        d = (s - target).norm(dim=-1).mean().item()
        assert d < d_plain, (name, d)
        print(f"{name}: {d:.3f}")

    # 5. Smoothing with sigma = 0 and eps_bsz > 1 equals no smoothing (all copies identical).
    a = TFG(eps_fn, log_p_fn, abar, TFGConfig(sigma=0.0, eps_bsz=1, clip_x0=0)).sample(n, (D,), 10, x_init=x_init)
    b = TFG(eps_fn, log_p_fn, abar, TFGConfig(sigma=0.0, eps_bsz=4, clip_x0=0)).sample(n, (D,), 10, x_init=x_init)
    assert torch.allclose(a, b, atol=1e-5)
    print("sigma = 0: eps_bsz has no effect, as it should")

    # 6. Schedules integrate to the base value.
    tf = TFG(eps_fn, log_p_fn, abar, TFGConfig())
    idx = torch.linspace(K - 1, 0, 50).long()
    ap = abar[idx]; app = torch.cat([abar[idx[1:]], torch.ones(1)])
    rhos = torch.stack([tf._strength(1.0, "increase", i, ap, app) for i in range(50)])
    assert abs(rhos.mean().item() - 1.0) < 1e-5
    print(f"rho schedule: mean {rhos.mean():.4f}, first {rhos[0]:.3f}, last {rhos[-1]:.3f} ('increase')")

    # 7. rho = mu = 0 with eta = 1 draws exactly the noise plain ancestral sampling draws, so
    #    the two agree bit for bit given identically seeded generators (deviation 8).
    def plain_ancestral(x, steps, gen):
        idx = torch.linspace(K - 1, 0, steps).long()
        for i, k in enumerate(idx):
            a = abar[k]; a_prev = abar[idx[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            t = (k.float() / K).repeat(x.shape[0])
            eps = eps_fn(x, t)
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            sig = ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sig ** 2).clamp(min=0).sqrt() * eps
            x = x + sig * torch.randn(x.shape, generator=gen)
        return x
    off1 = TFG(eps_fn, log_p_fn, abar, TFGConfig(rho=0, mu=0, iter_steps=0, clip_x0=0, eta=1.0))
    a = off1.sample(n, (D,), 20, x_init=x_init, generator=torch.Generator().manual_seed(7))
    b = plain_ancestral(x_init.clone(), 20, torch.Generator().manual_seed(7))
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()
    print("rho = mu = 0 at eta = 1 reproduces plain ancestral sampling with the same generator")

    # 8. Per-dimension clamp (deviation 7): a tensor of bounds clamps each dimension to its own
    #    range, a (lo, hi) pair to an asymmetric one, and a saturated dimension gets no gradient.
    bounds = torch.full((D,), 0.5)
    tfc = TFG(eps_fn, log_p_fn, abar, TFGConfig(clip_x0=bounds))
    x0 = tfc._predict_x0(torch.randn(n, D) * 10, torch.zeros(n, D), abar[0])
    assert x0.abs().max() <= 0.5 + 1e-6
    lo, hi = torch.full((D,), -0.25), torch.full((D,), 1.0)
    x0 = TFG(eps_fn, log_p_fn, abar, TFGConfig(clip_x0=(lo, hi)))._predict_x0(torch.randn(n, D) * 10, torch.zeros(n, D), abar[0])
    assert x0.min() >= -0.25 - 1e-6 and x0.max() <= 1.0 + 1e-6
    xg = (torch.ones(n, D) * 10).requires_grad_(True)
    g = torch.autograd.grad(tfc._predict_x0(xg, torch.zeros(n, D), abar[0]).sum(), xg)[0]
    assert g.abs().max() == 0, "a saturated clamp must have zero gradient"
    # With a per-dimension clamp the guided sample cannot leave the box even under strong guidance.
    strong = TFG(eps_fn, log_p_fn, abar, TFGConfig(rho=50.0, mu=10.0, clip_x0=torch.full((D,), 3.0)))
    s = strong.sample(n, (D,), 20, x_init=x_init)
    assert s.isfinite().all()
    print(f"per-dimension clamp: bounds respected, saturated gradient zero, strong guidance stays finite "
          f"(max |x| {s.abs().max():.2f})")
