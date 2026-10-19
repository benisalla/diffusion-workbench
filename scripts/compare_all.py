"""Train and evaluate every method on the same data with the same budget (the README's results).

    python scripts/compare_all.py --list
    python scripts/compare_all.py --only flow-matching          # one run: train + NFE sweep + figures
    python scripts/compare_all.py                               # all runs, one after the other

Each run writes runs/<name>/ (last.pt, history.json, eval.json, samples.png, trajectory.png).
`scripts/results_table.py` then gathers them into assets/ (table + figures).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from diffusion import train as train_mod
from diffusion.evaluate import Evaluator, evaluate_checkpoint
from diffusion.sample import generate, load_checkpoint
from diffusion.viz import save_grid, save_trajectory

# name -> (diffusion-train arguments, primary sampler, secondary sampler)
RUNS = {
    "ddpm-eps": (["--method", "ddpm", "--prediction", "eps", "--schedule", "cosine"], "ddim", "ddpm"),
    "ddpm-x0": (["--method", "ddpm", "--prediction", "x0", "--schedule", "cosine"], "ddim", "ddpm"),
    "ddpm-v": (["--method", "ddpm", "--prediction", "v", "--schedule", "cosine"], "ddim", "ddpm"),
    "score-sde-vp": (["--method", "score-sde", "--sde", "vp"], "ode", "em"),
    "score-sde-ve": (["--method", "score-sde", "--sde", "ve"], "ode", "em"),
    "flow-matching": (["--method", "flow-matching"], "euler", "heun"),
    "edm": (["--method", "edm"], "heun", "euler"),
    "flow-matching-cfg": (["--method", "flow-matching", "--cond"], "euler", None),
    "flow-matching-dit": (["--method", "flow-matching", "--backbone", "dit", "--lr", "5e-4"], "euler", "heun"),
}
PRIMARY_NFES = (5, 10, 20, 50, 100)
SECONDARY_NFES = (10, 50)


def run(name: str, a) -> None:
    args, primary, secondary = RUNS[name]
    out = Path(a.runs_dir) / name
    ckpt = out / "last.pt"
    if not ckpt.exists() or a.retrain:
        cfg = train_mod.config_from_args(train_mod.build_parser().parse_args(
            args + ["--dataset", a.dataset, "--max-steps", str(a.steps), "--max-minutes", str(a.max_minutes),
                    "--threads", str(a.threads), "--output-dir", str(out)]))
        train_mod.train(cfg)
    torch.set_num_threads(a.threads)
    net, method, cfg = load_checkpoint(str(ckpt))
    k = cfg.model.num_classes
    ev = Evaluator(cfg.train.dataset)
    t0 = time.time()
    rows = evaluate_checkpoint(str(ckpt), ev, PRIMARY_NFES, [primary], a.n_samples)
    if secondary:
        rows += evaluate_checkpoint(str(ckpt), ev, SECONDARY_NFES, [secondary], a.n_samples)
    if k:  # classifier-free guidance: same sampler, NFE budget 20, several guidance scales
        for w in (0.0, 3.0):
            rows += evaluate_checkpoint(str(ckpt), ev, (20,), [primary], a.n_samples, guidance=w)
    labels = torch.arange(64) % k if k else None
    steps = method.steps_for_nfe(primary, 50)
    save_grid(generate(net, method, 64, steps, primary, labels), out / "samples.png", nrow=8)
    if k:
        lab = torch.arange(10).repeat_interleave(8)[:80] % k
        for w in (1.0, 3.0):
            save_grid(generate(net, method, 80, method.steps_for_nfe(primary, 20), primary, lab, guidance=w),
                      out / f"samples_w{w:g}.png", nrow=8)
    record: list = []
    generate(net, method, 6, method.steps_for_nfe(primary, 20), primary, labels[:6] if k else None, record=record)
    save_trajectory(record, out / "trajectory.png", title=f"{name}: {primary}, NFE 20")
    hist = json.loads((out / "history.json").read_text())
    (out / "eval.json").write_text(json.dumps({
        "name": name, "method": cfg.method.method, "backbone": cfg.model.backbone, "primary": primary,
        "secondary": secondary, "params": hist["params"], "steps": hist["steps"],
        "train_seconds": hist["train_seconds"], "eval_seconds": time.time() - t0, "n_samples": a.n_samples,
        "floor": ev.real_floor(a.n_samples), "rows": rows}, indent=1))
    print(f"{name}: done -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None, choices=list(RUNS))
    p.add_argument("--list", action="store_true")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--max-minutes", type=float, default=12.0)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--retrain", action="store_true")
    a = p.parse_args()
    if a.list:
        for name, (args, primary, secondary) in RUNS.items():
            print(f"{name:20s} {' '.join(args):70s} samplers: {primary}, {secondary}")
        return
    if a.threads == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    for name in a.only or RUNS:
        run(name, a)


if __name__ == "__main__":
    sys.exit(main())
