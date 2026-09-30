"""Frozen U-Net with a PixArt-alpha-inspired residual cross-attention adapter.

This is the *mechanism* from PixArt-alpha -- text enters through residual cross-attention
with a zero-initialised output projection -- attached to a small frozen model. It is not a
reproduction of PixArt-alpha the model.

PAPER
    Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang,
    James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li.
    "PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic
    Text-to-Image Synthesis." ICLR 2024. arXiv:2310.00426.

CODE PROVENANCE
    Official code: https://github.com/PixArt-alpha/PixArt-alpha (checked 2026-09-28).
    The relevant pieces are `MultiHeadCrossAttention` in
    diffusion/model/nets/PixArt_blocks.py and its use in `PixArtBlock.forward` and
    `PixArt.initialize_weights` in diffusion/model/nets/PixArt.py. We did not copy their
    file: it depends on xformers and on their DiT. TextCrossAttention below is a
    re-expression of the same block in plain PyTorch.

WHAT WE FOLLOW FROM THEIR CODE
    - Queries come from the image features, keys and values from the text tokens,
      through separate `q_linear` and `kv_linear` projections and one output `proj`.
    - Padding tokens are masked out of the attention.
    - The block is added as a plain residual, x + cross_attn(x, text), with no norm and
      no gate -- exactly the line in PixArtBlock.forward.
    - The output projection `proj` is zero-initialised (initialize_weights), so the
      block is an exact no-op when first attached and the model starts identical to
      the frozen base.

WHAT WE CHANGED, AND WHY
    1. Backbone. Theirs is a DiT with cross-attention in every transformer block. Ours
       is a small convolutional U-Net with no attention anywhere. We attach ONE block at
       the bottleneck, flattening the (C, H, W) feature map into H*W query tokens and
       reshaping back afterwards.
    2. Text encoder and projection. Theirs is T5-XXL (4096-dim), passed through a
       caption-embedding MLP (`y_embedder`) before any block sees it. Ours is the CLIP
       text encoder (512-dim), the same frozen encoder every other arm in this project
       uses, so the comparison isolates the mechanism rather than the encoder -- and the
       tokens go straight into `kv_linear` with no intermediate MLP. `kv_linear`
       therefore projects from text_dim to 2 * channels rather than from channels.
    3. Attention kernel. Theirs uses xformers memory-efficient attention with a
       block-diagonal mask. Ours uses torch.nn.functional.scaled_dot_product_attention
       with a boolean key-padding mask. Same computation.
    4. Training. Theirs trains the whole model from scratch in stages. Ours freezes the
       base diffusion model and trains only the cross-attention block, so the base and
       the text-conditioned model share every other weight. That is the same discipline
       as the TC-LoRA arm and it is what makes the comparison a controlled one.
    5. Scale. PixArt-alpha is ~600M parameters trained on ~25M images. Ours adds ~0.4M
       parameters (394,240 at channels=256, text_dim=512) to a 1.9M model on 45k 32x32
       images.
    6. No classifier-free guidance at sampling. PixArt-alpha reports its results with
       CFG (scale 4.5). We sample purely conditionally, so this arm measures what the
       conditioning mechanism does on its own, without a guidance rule layered on top.
       That is a choice for this comparison, not their practice.
    7. Vector backbones. For the cell-state MLP there is no token axis: the hidden vector
       (B, C) is treated as ONE query token attending over the text tokens, and the
       block's output is added back to that vector. The softmax is over the text tokens,
       so it still selects between words; only the query side is degenerate. Attached
       after the MLP's input projection (`attach="inp"`, C = 1024).

BOOKKEEPING NOTE
    TextConditioned inserts the adapter INSIDE the backbone (setattr on the named stage),
    so after attaching, `backbone.parameters()` includes the adapter. To count frozen
    parameters, filter on `not p.requires_grad`; counting all of `backbone.parameters()`
    overstates the frozen model by exactly the adapter size.

TOKEN EMBEDDINGS, NOT POOLED
    Cross-attention over a single pooled vector is degenerate: with one key, the softmax
    is identically 1 and the block reduces to adding a projected constant. It still
    conditions, but it cannot select between words, which is the whole point of the
    mechanism. Feed the per-token embeddings and the padding mask.

Run this file directly to execute the self-checks at the bottom.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCrossAttention(nn.Module):
    """PixArt-alpha's MultiHeadCrossAttention, re-expressed for a conv feature map.

    x:    (B, C, H, W) feature map, or (B, N, C) tokens.
    text: (B, L, text_dim) token embeddings from a frozen text encoder.
    mask: (B, L) bool, True where the token is real (not padding).
    """

    def __init__(self, channels: int, text_dim: int, heads: int = 4):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        self.channels, self.heads = channels, heads
        self.q_linear = nn.Linear(channels, channels)
        self.kv_linear = nn.Linear(text_dim, 2 * channels)
        self.proj = nn.Linear(channels, channels)
        # PixArt.initialize_weights: zero the output projection so the block starts
        # as an exact identity.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, text, mask=None):
        is_map = x.dim() == 4
        is_vec = x.dim() == 2
        if is_map:
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        elif is_vec:
            tokens = x[:, None, :]                            # (B, 1, C): one query token (deviation 7)
        else:
            tokens = x
        b, n, c = tokens.shape
        hd = c // self.heads

        q = self.q_linear(tokens).view(b, n, self.heads, hd).transpose(1, 2)      # (B,h,N,hd)
        k, v = self.kv_linear(text).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, hd).transpose(1, 2)                        # (B,h,L,hd)
        v = v.view(b, -1, self.heads, hd).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]                # (B,1,1,L); True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.proj(out.transpose(1, 2).reshape(b, n, c))

        out = tokens + out                                    # PixArtBlock: x + cross_attn(x, y)
        if is_map:
            out = out.transpose(1, 2).reshape(b, c, h, w)
        elif is_vec:
            out = out[:, 0, :]
        return out


class _Conditioned(nn.Module):
    """Wraps one stage of the backbone: run it, then cross-attend to the text."""

    def __init__(self, inner: nn.Module, channels: int, text_dim: int, heads: int):
        super().__init__()
        self.inner = inner
        self.attn = TextCrossAttention(channels, text_dim, heads)
        self.text = None
        self.mask = None

    def forward(self, x):
        h = self.inner(x)
        if self.text is None:
            return h
        return self.attn(h, self.text, self.mask)


class TextConditioned(nn.Module):
    """Frozen backbone plus one trainable cross-attention block.

        model = TextConditioned(unet, attach="mid", channels=256, text_dim=512)
        eps   = model(x_t, t, text=tokens, mask=mask, cond_ids=labels)

    `attach` names the backbone attribute whose output the block conditions on.
    Everything after `mask` is forwarded to the backbone unchanged, so the label path
    can be kept (cond_ids=labels) or switched off (cond_ids=null) by the caller.
    """

    def __init__(self, backbone: nn.Module, attach: str, channels: int, text_dim: int,
                 heads: int = 4):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.attach = attach
        block = _Conditioned(getattr(backbone, attach), channels, text_dim, heads)
        setattr(backbone, attach, block)
        self.block = block

    def forward(self, x, t, text, mask=None, *args, **kwargs):
        self.block.text, self.block.mask = text, mask
        try:
            return self.backbone(x, t, *args, **kwargs)
        finally:
            self.block.text = self.block.mask = None

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    class ToyUNet(nn.Module):
        def __init__(self, c=16):
            super().__init__()
            self.enc = nn.Conv2d(3, c, 3, padding=1)
            self.mid = nn.Sequential(nn.Conv2d(c, 2 * c, 3, padding=1), nn.SiLU())
            self.dec = nn.Conv2d(2 * c, 3, 3, padding=1)

        def forward(self, x, t, cond_ids=None):
            return self.dec(self.mid(F.silu(self.enc(x))))

    base = ToyUNet()
    x, t = torch.randn(4, 3, 16, 16), torch.rand(4)
    text = torch.randn(4, 7, 32)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[:, 5:] = False                                     # two padding tokens
    ref = base(x, t).detach().clone()                       # before attaching

    model = TextConditioned(base, attach="mid", channels=32, text_dim=32, heads=4)

    out0 = model(x, t, text, mask)
    assert torch.allclose(out0, ref, atol=1e-6), (out0 - ref).abs().max().item()
    print("no-op at init: zero-initialised proj leaves the frozen backbone unchanged")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(".attn." in n for n in trainable), sorted(trainable)[:3]
    print(f"{model.n_trainable():,} trainable params, all in the cross-attention block")

    model(x, t, text, mask).sum().backward()
    g = model.block.attn.proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "proj received no gradient"
    print("proj has nonzero gradient, block is trainable")

    with torch.no_grad():
        model.block.attn.proj.weight.normal_(0, 0.05)
        a = model(x, t, text, mask)
        b = model(x, t, torch.randn_like(text), mask)       # different words
        rel = ((a - b).norm() / a.norm()).item()
    assert rel > 0.01, f"text is ignored (rel diff {rel:.4f})"
    print(f"output changes by {rel:.1%} when the text changes")

    with torch.no_grad():
        text2 = text.clone(); text2[:, 5:] = 99.0           # corrupt only padding tokens
        c = model(x, t, text2, mask)
    assert torch.allclose(a, c, atol=1e-5), "padding tokens leaked into attention"
    print("padding tokens are masked out")

    # The degenerate case from the header: with ONE key token the softmax is exactly 1
    # everywhere, so the attention output is the projected value regardless of the query.
    with torch.no_grad():
        single, m1 = text[:, :1], torch.ones(4, 1, dtype=torch.bool)
        blk = model.block.attn
        _, v = blk.kv_linear(single).chunk(2, dim=-1)
        expected = blk.proj(v)                              # (B, 1, C), same for every query
        q_tokens = torch.randn(4, 9, 32)                    # arbitrary queries
        got = blk(q_tokens, single, m1) - q_tokens          # residual removed
    assert torch.allclose(got, expected.expand(-1, 9, -1), atol=1e-5)
    print("one text token collapses attention to a constant, as the header warns")
