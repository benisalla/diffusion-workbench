"""05 - Every method, same data, same network, same budget (about 1 minute per method).

A small version of scripts/compare_all.py on the offline shapes data: trains each formulation for
`--steps` steps, then samples 100 images at NFE 20 with its default sampler and scores them with
the classifier-feature FID. Prints one table and saves one grid per method.

    python examples/05_compare_all_methods.py --steps 100
"""

import argparse
import time
from pathlib import Path

from diffusion.config import MethodConfig, RunConfig
from diffusion.data import load_dataset
from diffusion.evaluate import Evaluator, train_classifier
from diffusion.methods import method_label
from diffusion.sample import generate
from diffusion.train import train
from diffusion.viz import save_grid

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100)
a = p.parse_args()

METHODS = [MethodConfig("ddpm", "eps", "cosine"), MethodConfig("ddpm", "v", "cosine"),
           MethodConfig("score-sde", sde="vp"), MethodConfig("score-sde", sde="ve"),
           MethodConfig("flow-matching"), MethodConfig("edm")]
out = Path("examples/output/compare")
data = load_dataset("shapes", train=True, n=4000)
test = load_dataset("shapes", train=False)
clf, _ = train_classifier("shapes", steps=300, data=(data[:2], test[:2], 4), verbose=False)
ev = Evaluator("shapes", clf=clf, test=test)
print(f"{'method':16s} {'final loss':>10s} {'train':>7s} {'FID@20':>7s}   (floor {ev.real_floor(100):.2f})")
for mc in METHODS:
    cfg = RunConfig(method=mc)
    name = method_label(mc)
    cfg.train.max_steps, cfg.train.log_every, cfg.train.output_dir = a.steps, 25, str(out / name)
    t0 = time.time()
    res = train(cfg, data=data, verbose=False)
    method = res["method"]
    sampler = method.samplers[0]
    x = generate(res["ema"], method, 100, method.steps_for_nfe(sampler, 20), sampler)
    save_grid(x[:64], out / f"{name}.png", title=f"{name} ({sampler}, NFE 20)")
    print(f"{name:16s} {res['history']['loss'][-1]:10.4f} {time.time() - t0:6.0f}s {ev.fid(x):7.2f}", flush=True)
print("grids ->", out)
