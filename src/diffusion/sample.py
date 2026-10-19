"""Load a checkpoint and generate images: `diffusion-sample`.

    diffusion-sample checkpoints/ddpm/last.pt --sampler ddim --steps 20 --out ddim20.png
    diffusion-sample checkpoints/cfg/last.pt --labels all --guidance 3 --out cfg.png
    diffusion-sample checkpoints/flow/last.pt --steps 10 --trajectory traj.png
"""

from __future__ import annotations

import argparse
import time

import torch

from diffusion.backbones import build_backbone
from diffusion.config import RunConfig
from diffusion.methods import build_method


def load_checkpoint(path: str, device: str = "cpu", weights: str = "ema"):
    """-> (network in eval mode, method, RunConfig). `weights` = "ema" (default) or "net"."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = RunConfig.from_dict(ckpt["config"])
    net = build_backbone(cfg.model)
    net.load_state_dict(ckpt[weights])
    return net.to(device).eval(), build_method(cfg.method), cfg


class NFECounter(torch.nn.Module):
    """Wraps a network and counts how many times it is called per sample (CFG calls count once)."""

    def __init__(self, net):
        super().__init__()
        self.net, self.num_classes, self.calls = net, net.num_classes, 0
        self.in_channels, self.image_size = net.in_channels, net.image_size

    def forward(self, x, t, y=None):
        self.calls += 1
        return self.net(x, t, y)


@torch.no_grad()
def generate(net, method, n: int, steps: int, sampler: str | None = None, labels: torch.Tensor | None = None,
             guidance: float = 1.0, seed: int = 0, batch_size: int = 500, record: list | None = None,
             shape: tuple | None = None) -> torch.Tensor:
    """n images in [-1, 1], generated in batches. `record` (a list) collects the trajectory of the first batch."""
    device = next(net.parameters()).device
    if shape is None:
        shape = (net.in_channels, net.image_size, net.image_size)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        y = labels[start:start + b].to(device) if labels is not None else None
        out.append(method.sample(net, b, steps, sampler, y=y, guidance=guidance, shape=shape, generator=gen,
                                 record=record if start == 0 else None).clamp(-1, 1).cpu())
    return torch.cat(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate images from a trained checkpoint.")
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--sampler", default=None, help="ddpm: ddim | ddpm; score-sde: em | ode; flow-matching: euler | "
                                                     "heun; edm: heun | euler (default: the first one)")
    p.add_argument("--labels", default=None, help="'all' = 0..K-1 repeated, or one class id (conditional models)")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance scale w (1 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=("ema", "net"), default="ema")
    p.add_argument("--out", default="samples.png")
    p.add_argument("--trajectory", default=None, help="also save the denoising trajectory figure here")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, method, cfg = load_checkpoint(a.checkpoint, device, a.weights)
    labels = None
    if a.labels is not None:
        if not cfg.model.num_classes:
            p.error("this checkpoint is unconditional (trained without --cond)")
        k = cfg.model.num_classes
        labels = torch.arange(a.n) % k if a.labels == "all" else torch.full((a.n,), int(a.labels))
    from diffusion.viz import save_grid, save_trajectory

    counter = NFECounter(net)
    record = [] if a.trajectory else None
    t0 = time.time()
    imgs = generate(counter, method, a.n, a.steps, a.sampler, labels, a.guidance, a.seed, record=record)
    dt = time.time() - t0
    nrow = cfg.model.num_classes if (labels is not None and a.labels == "all") else 8
    save_grid(imgs, a.out, nrow=nrow)
    nfe = counter.calls // -(-a.n // 500)  # calls per batch = network evaluations per sample
    print(f"{a.n} images, {a.steps} steps, NFE/sample {nfe}, {dt:.1f} s -> {a.out}")
    if record:
        save_trajectory(record, a.trajectory, title=f"{cfg.method.method} {a.sampler or ''} {a.steps} steps")
        print("trajectory ->", a.trajectory)


if __name__ == "__main__":
    main()
