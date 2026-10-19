"""01 - The forward process of every method, side by side (no training, a few seconds).

Each formulation turns a clean image x0 into noise in its own way. This script takes the same eight
images and draws x_t at six noise levels for: DDPM (linear and cosine schedules), the VP and VE SDEs,
the flow-matching straight line and EDM's x0 + sigma * noise. Look at:
  * the linear DDPM schedule destroys the image much earlier than the cosine one,
  * the VE SDE never shrinks the image, it only adds noise up to sigma = 50 (drawn clipped to [-1, 1]),
  * flow matching fades the image out linearly while the noise fades in.

    python examples/01_forward_process.py          -> examples/output/forward_process.png
"""

from pathlib import Path

import torch

from diffusion.data import load_dataset
from diffusion.methods import DDPM, EDM, ScoreSDE
from diffusion.viz import save_grid

OUT = Path("examples/output")
torch.manual_seed(0)
x0, _, _ = load_dataset("shapes", train=False, n=8)
eps = torch.randn_like(x0)
levels = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])  # fraction of the way from data to noise

rows, names = [], []
for schedule in ("linear", "cosine"):
    m = DDPM(1000, schedule)
    for u in levels:
        t = torch.full((8,), int(u * 999))
        rows.append(m.q_sample(x0, t, eps))
    names.append(f"DDPM {schedule}: x_t = sqrt(abar) x0 + sqrt(1 - abar) eps")
for kind in ("vp", "ve"):
    sde = ScoreSDE(kind)
    for u in levels:
        m, s = sde.marginal(torch.full((8,), max(u.item(), 1e-3)))
        rows.append(m.reshape(-1, 1, 1, 1) * x0 + s.reshape(-1, 1, 1, 1) * eps)
    names.append(f"score SDE {kind.upper()}: x_t = m(t) x0 + s(t) eps")
for u in levels:
    rows.append((1 - u) * x0 + u * eps)
names.append("flow matching: x_t = (1 - t) x0 + t eps")
edm = EDM()
for sigma in edm.sigmas(5).flip(0):  # 0, then the 5-step Karras schedule up to sigma_max = 80
    rows.append(x0 + sigma * eps)
names.append("EDM: x = x0 + sigma eps, sigma on the Karras schedule")

# one figure: for each method, 8 images (rows of the grid) x 6 noise levels (columns)
grid = torch.stack(rows).reshape(len(names), len(levels), 8, 1, 28, 28).permute(0, 2, 1, 3, 4, 5)
OUT.mkdir(parents=True, exist_ok=True)
for i, name in enumerate(names):
    save_grid(grid[i, :3].reshape(-1, 1, 28, 28), OUT / f"forward_{i}.png", nrow=len(levels), title=name)
    print(f"{name:65s} -> {OUT / f'forward_{i}.png'}")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

short = ["DDPM linear", "DDPM cosine", "SDE VP", "SDE VE", "flow matching", "EDM"]
fig, axes = plt.subplots(len(names), len(levels), figsize=(len(levels) * 1.1, len(names) * 1.15))
for i in range(len(names)):
    for j in range(len(levels)):
        ax = axes[i, j]
        ax.imshow(((grid[i, 0, j, 0].clamp(-1, 1) + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(short[i], fontsize=7)
        if i == 0:
            ax.set_title(f"{levels[j].item():.2g}", fontsize=7)
fig.suptitle("forward process: fraction of the way from data to noise", fontsize=8)
fig.savefig(OUT / "forward_process.png", bbox_inches="tight", dpi=120)
print("all methods, one image ->", OUT / "forward_process.png")

# the signal-to-noise ratio is what really differs between schedules
for schedule in ("linear", "cosine"):
    a = DDPM(1000, schedule).abar
    print(f"DDPM {schedule:6s}: SNR = abar / (1 - abar) at t = 100, 500, 900: "
          + ", ".join(f"{(a[t] / (1 - a[t])).item():.3g}" for t in (100, 500, 900)))
