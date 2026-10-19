"""Figures: sample grids, denoising trajectories, loss curves, NFE-vs-quality curves.

matplotlib is imported lazily with the non-interactive Agg backend, so everything works headless.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_grid(x: torch.Tensor, nrow: int = 8, pad: int = 2) -> torch.Tensor:
    """(N, C, H, W) in [-1, 1] -> one (H', W', C) image in [0, 1]."""
    x = ((x.detach().float().clamp(-1, 1) + 1) / 2).cpu()
    n, c, h, w = x.shape
    rows = (n + nrow - 1) // nrow
    grid = torch.ones(c, rows * (h + pad) + pad, nrow * (w + pad) + pad)
    for i in range(n):
        r, k = divmod(i, nrow)
        grid[:, pad + r * (h + pad): pad + r * (h + pad) + h, pad + k * (w + pad): pad + k * (w + pad) + w] = x[i]
    return grid.permute(1, 2, 0)


def save_grid(x: torch.Tensor, path: str | Path, nrow: int = 8, title: str | None = None) -> None:
    plt = _plt()
    g = make_grid(x, nrow)
    fig, ax = plt.subplots(figsize=(g.shape[1] / 40, g.shape[0] / 40 + (0.3 if title else 0)), dpi=100)
    ax.imshow((g.squeeze(-1) if g.shape[-1] == 1 else g).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_trajectory(record: list, path: str | Path, n_show: int = 6, n_cols: int = 10, title: str | None = None):
    """Two rows per image: the current x_t along the reverse process, and the network's guess of x0."""
    plt = _plt()
    idx = torch.linspace(0, len(record) - 1, min(n_cols, len(record))).round().long().tolist()
    n_show = min(n_show, record[0][1].shape[0])
    fig, axes = plt.subplots(2 * n_show, len(idx), figsize=(len(idx) * 0.9, 2 * n_show * 0.9 + 0.4), squeeze=False)
    for j, k in enumerate(idx):
        t, x, x0 = record[k]
        for i in range(n_show):
            for row, img in ((2 * i, x[i]), (2 * i + 1, x0[i])):
                ax = axes[row, j]
                im = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).squeeze(-1).numpy()
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel("x_t" if row % 2 == 0 else "x0_hat", fontsize=6)
        axes[0, j].set_title(f"{t:.3g}", fontsize=6)
    if title:
        fig.suptitle(title, fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_losses(histories: dict[str, dict], path: str | Path, title: str = "training loss") -> None:
    """histories: label -> {"step": [...], "loss": [...]}. Each method has its own loss scale, so we
    plot loss divided by its first logged value (relative progress) on a log axis."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, h in histories.items():
        loss = torch.tensor(h["loss"])
        ax.plot(h["step"], (loss / loss[0]).tolist(), label=label, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss / first logged loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_nfe_curves(curves: dict[str, list[tuple[int, float]]], path: str | Path, floor: float | None = None,
                    ylabel: str = "classifier-feature FID (lower is better)") -> None:
    """curves: label -> [(nfe, metric), ...]."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, pts in curves.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5, lw=1.4, label=label)
    if floor is not None:
        ax.axhline(floor, color="gray", ls="--", lw=1, label="real vs real (floor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("network evaluations per sample (NFE)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
