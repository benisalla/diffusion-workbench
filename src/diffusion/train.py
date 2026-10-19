"""One training loop for every method and backbone: `diffusion-train`.

    for step in range(max_steps):
        x0, y = random batch                     (labels replaced by "null" with prob p_uncond, for CFG)
        loss = method.loss(net, x0, y)           (the only line that depends on the method)
        AdamW step, gradient clipping, EMA of the weights

The exponential moving average (EMA) of the weights is what we sample from: it averages out the
noise of the last steps and gives visibly cleaner samples than the raw weights, especially in short runs.
Writes <output_dir>/last.pt (config + EMA weights + history), history.json, loss.png and samples.png.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch

from diffusion.backbones import build_backbone, count_parameters
from diffusion.config import BACKBONES, METHODS, RunConfig
from diffusion.data import DATASETS, load_dataset
from diffusion.methods import build_method


def setup(threads: int, seed: int, device: str = "auto") -> torch.device:
    if threads == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@torch.no_grad()
def ema_update(ema: torch.nn.Module, net: torch.nn.Module, decay: float) -> None:
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.lerp_(p, 1 - decay)


def train(cfg: RunConfig, data: tuple | None = None, verbose: bool = True) -> dict:
    """Train one model; returns {"net", "ema", "method", "history", "cfg", "path"}.

    `data` = (images, labels, num_classes) skips loading a dataset (used by tests and examples).
    """
    tc = cfg.train
    device = setup(tc.threads, tc.seed, tc.device)
    x, y, n_classes = data if data is not None else load_dataset(tc.dataset, tc.data_dir, True, n=tc.n_train,
                                                                 seed=tc.seed)
    cfg.model.in_channels, cfg.model.image_size = x.shape[1], x.shape[2]
    cfg.model.num_classes = n_classes if tc.cond else 0
    net = build_backbone(cfg.model).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    method = build_method(cfg.method)
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=0.0)
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, tc.warmup_steps)))
    n_params = count_parameters(net)
    if verbose:
        print(f"{cfg.method.method} | {cfg.model.backbone} {n_params / 1e6:.2f} M params | {len(x)} images "
              f"{tuple(x.shape[1:])} | cond={tc.cond} | device {device}", flush=True)

    gen = torch.Generator().manual_seed(tc.seed)
    history = {"step": [], "loss": [], "time": []}
    t0, window = time.time(), []
    step = 0
    for step in range(1, tc.max_steps + 1):
        idx = torch.randint(0, len(x), (tc.batch_size,), generator=gen)
        xb = x[idx].to(device)
        yb = None
        if tc.cond:
            yb = y[idx].to(device)
            drop = torch.rand(len(yb), generator=gen).to(device) < tc.p_uncond
            yb = torch.where(drop, torch.full_like(yb, n_classes), yb)  # "null" label = n_classes
        net.train()
        loss = method.loss(net, xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip)
        opt.step()
        warm.step()
        ema_update(ema, net, min(tc.ema_decay, (1 + step) / (10 + step)))  # EMA warm-up for short runs
        window.append(loss.item())
        if step % tc.log_every == 0 or step == tc.max_steps:
            elapsed = time.time() - t0
            history["step"].append(step)
            history["loss"].append(sum(window) / len(window))
            history["time"].append(elapsed)
            window = []
            if verbose:
                print(f"step {step:6d}  loss {history['loss'][-1]:.4f}  {elapsed / 60:5.1f} min", flush=True)
        if tc.save_every and step % tc.save_every == 0:  # snapshots for "samples across training" figures
            snap = Path(tc.output_dir) / f"ckpt_{step:06d}.pt"
            snap.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": cfg.to_dict(), "ema": ema.state_dict(), "step": step}, snap)
        if tc.max_minutes and time.time() - t0 > tc.max_minutes * 60:
            if verbose:
                print(f"time limit reached at step {step}")
            break
    history["train_seconds"] = time.time() - t0
    history["steps"] = step
    history["params"] = n_params

    out = Path(tc.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "last.pt"
    torch.save({"config": cfg.to_dict(), "ema": ema.state_dict(), "net": net.state_dict(), "history": history},
               path)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    return {"net": net, "ema": ema, "method": method, "history": history, "cfg": cfg, "path": str(path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a diffusion / flow model (one loop for every method).")
    g = p.add_argument_group("method")
    g.add_argument("--method", choices=METHODS, default="ddpm")
    g.add_argument("--prediction", choices=("eps", "x0", "v"), default="eps", help="ddpm: what the network predicts")
    g.add_argument("--schedule", choices=("linear", "cosine"), default="linear", help="ddpm noise schedule")
    g.add_argument("--timesteps", type=int, default=1000, help="ddpm: T")
    g.add_argument("--sde", choices=("vp", "ve"), default="vp", help="score-sde: variance preserving / exploding")
    g = p.add_argument_group("network")
    g.add_argument("--backbone", choices=BACKBONES, default="unet")
    g.add_argument("--base-channels", type=int, default=16)
    g.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 2])
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--dit-dim", type=int, default=128)
    g.add_argument("--dit-depth", type=int, default=4)
    g.add_argument("--dit-heads", type=int, default=4)
    g.add_argument("--patch-size", type=int, default=4)
    g = p.add_argument_group("training")
    g.add_argument("--dataset", choices=DATASETS, default="mnist")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--n-train", type=int, default=0, help="use a random subset of this size (0 = all)")
    g.add_argument("--cond", action="store_true", help="class-conditional training with label dropout (for CFG)")
    g.add_argument("--p-uncond", type=float, default=0.1)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--max-steps", type=int, default=2000)
    g.add_argument("--max-minutes", type=float, default=0.0)
    g.add_argument("--ema-decay", type=float, default=0.999)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--save-every", type=int, default=0, help="also save EMA snapshots every N steps")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--device", default="auto")
    g.add_argument("--output-dir", default="checkpoints/run")
    g.add_argument("--sample-steps", type=int, default=50, help="steps for the samples.png written at the end")
    return p


def config_from_args(a) -> RunConfig:
    cfg = RunConfig()
    cfg.method.method, cfg.method.prediction, cfg.method.schedule = a.method, a.prediction, a.schedule
    cfg.method.timesteps, cfg.method.sde = a.timesteps, a.sde
    m = cfg.model
    m.backbone, m.base_channels, m.channel_mults, m.dropout = (a.backbone, a.base_channels,
                                                              tuple(a.channel_mults), a.dropout)
    m.dit_dim, m.dit_depth, m.dit_heads, m.patch_size = a.dit_dim, a.dit_depth, a.dit_heads, a.patch_size
    t = cfg.train
    for k in ("dataset", "data_dir", "n_train", "cond", "p_uncond", "batch_size", "lr", "max_steps", "max_minutes",
              "ema_decay", "log_every", "save_every", "seed", "threads", "device", "output_dir"):
        setattr(t, k, getattr(a, k))
    return cfg


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    res = train(cfg)
    from diffusion.sample import generate
    from diffusion.viz import plot_losses, save_grid

    out = Path(cfg.train.output_dir)
    plot_losses({cfg.method.method: res["history"]}, out / "loss.png")
    labels = torch.arange(64) % cfg.model.num_classes if cfg.model.num_classes else None
    imgs = generate(res["ema"], res["method"], 64, a.sample_steps, labels=labels, guidance=1.0)
    save_grid(imgs, out / "samples.png", nrow=8)
    print(f"done in {res['history']['train_seconds'] / 60:.1f} min -> {res['path']}, samples.png, loss.png")


if __name__ == "__main__":
    main()
