"""Regenerate the "before and after training" figures of the README from saved checkpoints.

    python scripts/make_gifs.py                       # all three
    python scripts/make_gifs.py --only reverse        # before_after | reverse | progress

Writes to assets/:
  before_after.png       the same starting noise through the untrained and the trained network, per method
  reverse_process.gif    noise -> digits step by step: DDPM (ancestral), DDIM, flow matching, EDM side by side
  training_progress.gif  samples from one fixed noise as training goes on (snapshots every 25 steps)

Needs the checkpoints of `scripts/compare_all.py` (runs/<name>/last.pt) and, for the progress GIF, a
run trained with snapshots:

    diffusion-train --method flow-matching --max-steps 500 --save-every 25 --output-dir runs/flow-matching-progress
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusion.backbones import build_backbone
from diffusion.sample import generate, load_checkpoint
from diffusion.viz import make_grid

# (run, sampler, steps, label): DDPM and DDIM use the same x0-prediction checkpoint (the best DDPM run)
REVERSE = [
    ("ddpm-x0", "ddpm", 100, "DDPM ancestral, 100 steps"),
    ("ddpm-x0", "ddim", 50, "DDIM, 50 steps"),
    ("flow-matching", "euler", 50, "flow matching, Euler 50"),
    ("edm", "heun", 25, "EDM, Heun 25 (49 NFE)"),
]
BEFORE_AFTER = [
    ("ddpm-x0", "ddim", 50, "DDPM (x0-pred), DDIM 50", None),
    ("flow-matching", "euler", 50, "flow matching, Euler 50", None),
    ("edm", "heun", 25, "EDM, Heun 49 NFE", None),
    ("flow-matching-cfg", "euler", 20, "flow matching + CFG w=3", 3.0),
]


def untrained_like(net, cfg):
    """The same architecture with the initial weights of the training run (same seed)."""
    torch.manual_seed(cfg.train.seed)
    fresh = build_backbone(cfg.model).eval()
    return fresh


def to_pil(x: torch.Tensor, nrow: int, scale: int = 2) -> Image.Image:
    g = (make_grid(x, nrow).squeeze(-1).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(g)  # 2-d -> grayscale "L", (H, W, 3) -> RGB
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def caption(img: Image.Image, text: str, height: int = 18) -> Image.Image:
    out = Image.new(img.mode, (img.width, img.height + height), 255 if img.mode == "L" else (255, 255, 255))
    out.paste(img, (0, height))
    ImageDraw.Draw(out).text((4, 3), text, fill=0 if img.mode == "L" else (0, 0, 0))
    return out


def hstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new(images[0].mode, (w, h), 255)
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    w = max(i.width for i in images)
    out = Image.new(images[0].mode, (w, sum(i.height for i in images) + gap * (len(images) - 1)), 255)
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def save_gif(frames: list[Image.Image], path: Path, ms: int, hold: int = 12) -> None:
    frames = frames + [frames[-1]] * hold  # pause on the final images before looping
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"{path}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


def labels_for(cfg, n):
    return torch.arange(n) % cfg.model.num_classes if cfg.model.num_classes else None


def before_after(runs: Path, out: Path, n: int, seed: int) -> None:
    rows = []
    for run, sampler, steps, label, w in BEFORE_AFTER:
        ckpt = runs / run / "last.pt"
        if not ckpt.exists():
            print("skip (missing):", ckpt)
            continue
        net, method, cfg = load_checkpoint(str(ckpt))
        y, g = labels_for(cfg, n), w or 1.0
        before = generate(untrained_like(net, cfg), method, n, steps, sampler, y, g, seed=seed)
        after = generate(net, method, n, steps, sampler, y, g, seed=seed)
        nrow = int(n ** 0.5)
        rows.append(hstack([caption(to_pil(before, nrow), f"{label}: untrained"),
                            caption(to_pil(after, nrow), "after 500 steps (same noise)")], gap=16))
    img = vstack(rows, gap=10)
    img.save(out / "before_after.png", optimize=True)
    print(out / "before_after.png")


def reverse_process(runs: Path, out: Path, n: int, seed: int, n_frames: int = 24) -> None:
    panels = []
    for run, sampler, steps, label in REVERSE:
        net, method, cfg = load_checkpoint(str(runs / run / "last.pt"))
        record: list = []
        generate(net, method, n, steps, sampler, labels_for(cfg, n), seed=seed, record=record)
        # same number of frames for every method: frame f shows the state at fraction f of its steps
        idx = np.linspace(0, len(record) - 1, n_frames).round().astype(int)
        panels.append([caption(to_pil(record[i][1], int(n ** 0.5)), f"{label}  [{i + 1}/{len(record)}]")
                       for i in idx])
    frames = [hstack([p[f] for p in panels]) for f in range(n_frames)]
    save_gif(frames, out / "reverse_process.gif", ms=170, hold=6)


def training_progress(runs: Path, out: Path, n: int, seed: int) -> None:
    folder = runs / "flow-matching-progress"
    snaps = sorted(folder.glob("ckpt_*.pt"))
    if not snaps:
        raise SystemExit(f"no snapshots in {folder}: train with --save-every first (see the docstring)")
    net, method, cfg = load_checkpoint(str(snaps[0]))
    nrow = int(n ** 0.5)
    frames = [caption(to_pil(generate(untrained_like(net, cfg), method, n, 20, "euler", seed=seed), nrow, 3),
                      "step 0 (untrained)")]
    for snap in snaps:
        net, method, _ = load_checkpoint(str(snap))
        step = torch.load(snap, map_location="cpu", weights_only=False)["step"]
        x = generate(net, method, n, 20, "euler", seed=seed)
        frames.append(caption(to_pil(x, nrow, 3), f"training step {step}  (64 images/step)"))
    save_gif(frames, out / "training_progress.gif", ms=350, hold=6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--assets", default="assets")
    p.add_argument("--only", nargs="+", choices=["before_after", "reverse", "progress"],
                   default=["before_after", "reverse", "progress"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(1)
    runs, out = Path(a.runs_dir), Path(a.assets)
    out.mkdir(exist_ok=True)
    if "before_after" in a.only:
        before_after(runs, out, 16, a.seed)
    if "reverse" in a.only:
        reverse_process(runs, out, 16, a.seed)
    if "progress" in a.only:
        training_progress(runs, out, 36, a.seed)


if __name__ == "__main__":
    main()
