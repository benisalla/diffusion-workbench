"""02 - Train one method on the shapes data and sample from it (2-5 minutes on a CPU).

    python examples/02_train_one_method.py                         # DDPM, epsilon prediction
    python examples/02_train_one_method.py --method flow-matching
    python examples/02_train_one_method.py --method edm
    python examples/02_train_one_method.py --method score-sde --sde ve
    python examples/02_train_one_method.py --method ddpm --prediction v --schedule cosine
    python examples/02_train_one_method.py --method flow-matching --backbone dit

Every method uses the same small U-Net (or DiT), the same data, the same loop: only
`method.loss` and `method.sample` change. Writes examples/output/<name>/: last.pt (used by 03),
samples.png and trajectory.png (x_t and the network's guess of x0 along the reverse process).
"""

import argparse
from pathlib import Path

from diffusion.config import RunConfig
from diffusion.data import load_dataset
from diffusion.methods import method_label
from diffusion.sample import generate
from diffusion.train import train
from diffusion.viz import plot_losses, save_grid, save_trajectory

p = argparse.ArgumentParser()
p.add_argument("--method", default="ddpm", choices=["ddpm", "score-sde", "flow-matching", "edm"])
p.add_argument("--prediction", default="eps", choices=["eps", "x0", "v"])
p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
p.add_argument("--sde", default="vp", choices=["vp", "ve"])
p.add_argument("--backbone", default="unet", choices=["unet", "dit"])
p.add_argument("--steps", type=int, default=200)
a = p.parse_args()

cfg = RunConfig()
cfg.method.method, cfg.method.prediction, cfg.method.schedule, cfg.method.sde = (a.method, a.prediction,
                                                                                 a.schedule, a.sde)
cfg.model.backbone = a.backbone
cfg.train.max_steps, cfg.train.log_every = a.steps, 25
if a.backbone == "dit":
    cfg.train.lr = 5e-4
name = method_label(cfg.method) + ("-dit" if a.backbone == "dit" else "")
out = Path("examples/output") / name
cfg.train.output_dir = str(out)

data = load_dataset("shapes", train=True, n=4000)  # offline: generated, no download
res = train(cfg, data=data)
net, method = res["ema"], res["method"]
plot_losses({name: res["history"]}, out / "loss.png")

sampler = method.samplers[0]
steps = method.steps_for_nfe(sampler, 30)
save_grid(generate(net, method, 64, steps, sampler), out / "samples.png", title=f"{name}, {sampler}, NFE 30")
record = []
generate(net, method, 6, steps, sampler, record=record)
save_trajectory(record, out / "trajectory.png", title=f"{name}: reverse process ({sampler})")
print(f"{name}: {res['history']['steps']} steps in {res['history']['train_seconds'] / 60:.1f} min, "
      f"final loss {res['history']['loss'][-1]:.4f}\n -> {out}/samples.png, trajectory.png, loss.png, last.pt")
