"""03 - Same trained model, different samplers and step counts: quality vs cost (1-3 minutes).

Needs a checkpoint from example 02 (default: the DDPM one). Trains a small shapes classifier
(30 s), then scores every sampler of the method at 5, 10, 20 and 50 network evaluations (NFE)
with the classifier-feature FID (lower is better; see docs/evaluation.md for what it is and is not).

    python examples/02_train_one_method.py && python examples/03_samplers_and_nfe.py
    python examples/03_samplers_and_nfe.py --checkpoint examples/output/flow-matching/last.pt
"""

import argparse
from pathlib import Path

import torch

from diffusion.data import load_dataset
from diffusion.evaluate import Evaluator, evaluate_checkpoint, train_classifier
from diffusion.sample import generate, load_checkpoint
from diffusion.viz import plot_nfe_curves, save_grid

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="examples/output/ddpm-eps-linear/last.pt")
p.add_argument("--n-samples", type=int, default=100)
a = p.parse_args()
if not Path(a.checkpoint).exists():
    raise SystemExit(f"{a.checkpoint} not found: run examples/02_train_one_method.py first")

train_xy, test_xy = load_dataset("shapes", train=True, n=4000)[:2], load_dataset("shapes", train=False)
clf, acc = train_classifier("shapes", steps=300, data=(train_xy, test_xy[:2], 4), verbose=False)
print(f"shapes classifier: test accuracy {acc:.1%}")
ev = Evaluator("shapes", clf=clf, test=test_xy)
print(f"floor ({a.n_samples} real images vs test set): FID {ev.real_floor(a.n_samples):.2f}")

rows = evaluate_checkpoint(a.checkpoint, ev, nfes=(5, 10, 20, 50), n_samples=a.n_samples)
out = Path(a.checkpoint).parent
curves = {}
for r in rows:
    curves.setdefault(r["sampler"], []).append((r["nfe"], r["fid"]))
plot_nfe_curves(curves, out / "nfe.png")

net, method, _ = load_checkpoint(a.checkpoint)
row_imgs = []
for sampler in method.samplers:
    for nfe in (5, 10, 20, 50):
        row_imgs.append(generate(net, method, 8, method.steps_for_nfe(sampler, nfe), sampler, seed=1))
save_grid(torch.cat(row_imgs), out / "samplers.png", nrow=8,
          title=" / ".join(method.samplers) + ": one row per NFE 5, 10, 20, 50 (same noise)")
print(f"-> {out}/nfe.png, {out}/samplers.png")
