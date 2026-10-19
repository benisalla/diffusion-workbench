"""The two denoising networks shared by every method: a small U-Net and a small DiT.

Every method in `methods.py` calls the network the same way, `net(x, t, y)`:

    x : (B, C, H, W)  the noisy image (possibly rescaled by the method)
    t : (B,)          a float "time / noise level" already mapped by the method to about [0, 1000]
    y : (B,) or None  class labels; label `num_classes` is the "null" label used by classifier-free guidance

and gets back a (B, C, H, W) tensor. What that tensor *means* (noise, clean image, velocity, ...) is
decided by the method, not by the network: that is why one network serves all formulations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from diffusion.config import ModelConfig


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal features of a float time, like Transformer positions: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class Conditioning(nn.Module):
    """time (+ optional class label) -> one conditioning vector per image."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))
        # one extra row: the "null" label used for unconditional predictions in CFG
        self.label = nn.Embedding(num_classes + 1, dim * 4) if num_classes > 0 else None

    def forward(self, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        emb = self.mlp(timestep_embedding(t, self.dim))
        if self.label is not None:
            if y is None:
                y = torch.full((t.shape[0],), self.num_classes, device=t.device, dtype=torch.long)
            emb = emb + self.label(y)
        return emb


# --------------------------------------------------------------------------------------------- U-Net
class ResBlock(nn.Module):
    """GroupNorm-SiLU-conv twice, with the conditioning vector added in between (as in DDPM)."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Global self-attention over the pixels of a (small, 7 x 7) feature map."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Encoder-decoder with skip connections; one ResBlock per resolution, attention at the lowest one.

    28 x 28 (C) -> 28 (base) -> 14 (2 base) -> 7 (2 base, + attention) -> back up, concatenating skips.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_classes, self.in_channels, self.image_size = cfg.num_classes, cfg.in_channels, cfg.image_size
        base, mults = cfg.base_channels, cfg.channel_mults
        emb_dim = base * 4
        self.cond = Conditioning(base, cfg.num_classes)
        self.conv_in = nn.Conv2d(cfg.in_channels, base, 3, padding=1)
        chans = [base * m for m in mults]
        n = len(chans)
        self.down, self.down_attn, self.downsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c_prev = base
        for i, c in enumerate(chans):
            self.down.append(ResBlock(c_prev, c, emb_dim, cfg.dropout))
            self.down_attn.append(SelfAttention(c) if i == n - 1 else nn.Identity())
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < n - 1 else nn.Identity())
            c_prev = c
        self.mid1 = ResBlock(c_prev, c_prev, emb_dim, cfg.dropout)
        self.mid_attn = SelfAttention(c_prev)
        self.mid2 = ResBlock(c_prev, c_prev, emb_dim, cfg.dropout)
        self.up, self.up_attn, self.upsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(n)):
            c = chans[i]
            self.up.append(ResBlock(c_prev + c, c, emb_dim, cfg.dropout))
            self.up_attn.append(SelfAttention(c) if i == n - 1 else nn.Identity())
            self.upsample.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1))
                if i > 0 else nn.Identity()
            )
            c_prev = c
        self.norm_out = nn.GroupNorm(8, c_prev)
        self.conv_out = nn.Conv2d(c_prev, cfg.in_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # the network starts by predicting 0: a stable start for every target
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.cond(t, y)
        h = self.conv_in(x)
        skips = []
        for res, attn, down in zip(self.down, self.down_attn, self.downsample):
            h = attn(res(h, emb))
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for res, attn, up in zip(self.up, self.up_attn, self.upsample):
            h = attn(res(torch.cat([h, skips.pop()], dim=1), emb))
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ----------------------------------------------------------------------------------------------- DiT
class DiTBlock(nn.Module):
    """Transformer block whose LayerNorms are modulated by the conditioning (adaLN-Zero, Peebles & Xie 2023).

    The conditioning vector produces a shift, a scale and a gate for the attention and the MLP branch.
    The gates start at zero, so every block starts as the identity.
    """

    def __init__(self, dim: int, heads: int, cond_dim: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(cond_dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(F.silu(c))[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        b, n, d = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + gate1 * self.proj(h)
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class DiT(nn.Module):
    """Diffusion Transformer: cut the image into p x p patches, run Transformer blocks, paste patches back."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_classes, self.in_channels, self.image_size = cfg.num_classes, cfg.in_channels, cfg.image_size
        self.p, self.c = cfg.patch_size, cfg.in_channels
        self.side = cfg.image_size // cfg.patch_size
        dim = cfg.dit_dim
        self.cond = Conditioning(dim // 4, cfg.num_classes)  # outputs a vector of size dim
        self.patch = nn.Conv2d(cfg.in_channels, dim, cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.randn(1, self.side * self.side, dim) * 0.02)
        self.blocks = nn.ModuleList(DiTBlock(dim, cfg.dit_heads, dim) for _ in range(cfg.dit_depth))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, cfg.patch_size ** 2 * cfg.in_channels)
        for layer in (self.ada_out, self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        c = self.cond(t, y)
        h = self.patch(x).flatten(2).transpose(1, 2) + self.pos       # (B, N, dim), N = side^2
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.ada_out(F.silu(c))[:, None].chunk(2, dim=-1)
        h = self.out(self.norm_out(h) * (1 + scale) + shift)          # (B, N, p*p*C)
        b, s, p = x.shape[0], self.side, self.p
        h = h.reshape(b, s, s, p, p, self.c).permute(0, 5, 1, 3, 2, 4)  # (B, C, s, p, s, p)
        return h.reshape(b, self.c, s * p, s * p)


def build_backbone(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone == "unet":
        return UNet(cfg)
    if cfg.backbone == "dit":
        if cfg.image_size % cfg.patch_size:
            raise ValueError(f"image_size {cfg.image_size} is not divisible by patch_size {cfg.patch_size}")
        return DiT(cfg)
    raise ValueError(f"unknown backbone {cfg.backbone!r} (unet | dit)")


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
