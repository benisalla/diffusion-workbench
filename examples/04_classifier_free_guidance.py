"""04 - Class-conditional flow matching with classifier-free guidance (4-6 minutes).

Training: the label is given to the network, but replaced by a "null" label 10% of the time, so one
network learns both p(x | class) and p(x). Sampling: out = uncond + w * (cond - uncond).
w = 0 ignores the label, w = 1 is the plain conditional model, w > 1 exaggerates the class.
A shapes classifier measures how often a sample generated for "circle" really is a circle.

    python examples/04_classifier_free_guidance.py     -> examples/output/cfg/guidance.png
"""

from pathlib import Path

import torch

from diffusion.config import RunConfig
from diffusion.data import SHAPE_NAMES, load_dataset
from diffusion.evaluate import train_classifier
from diffusion.sample import generate
from diffusion.train import train
from diffusion.viz import save_grid

out = Path("examples/output/cfg")
cfg = RunConfig()
cfg.method.method = "flow-matching"
cfg.train.cond, cfg.train.p_uncond = True, 0.1
cfg.train.max_steps, cfg.train.log_every, cfg.train.output_dir = 250, 50, str(out)
data = load_dataset("shapes", train=True, n=4000)
res = train(cfg, data=data)
net, method = res["ema"], res["method"]

test = load_dataset("shapes", train=False)
clf, acc = train_classifier("shapes", steps=300, data=(data[:2], test[:2], 4), verbose=False)
print(f"shapes classifier accuracy on real test images: {acc:.1%}")

labels = torch.arange(4).repeat_interleave(25)  # 25 samples per shape
grids = []
for w in (0.0, 1.0, 3.0):
    x = generate(net, method, len(labels), 20, "euler", labels, guidance=w, seed=0)
    with torch.no_grad():
        hit = (clf(x).argmax(1) == labels).float().mean().item()
    print(f"guidance w = {w:g}: samples recognised as the requested shape {hit:.1%}")
    grids.append(x.reshape(4, 25, 1, 28, 28)[:, :8].reshape(-1, 1, 28, 28))
save_grid(torch.cat(grids), out / "guidance.png", nrow=8,
          title=f"rows: {', '.join(SHAPE_NAMES)}; blocks: w = 0, 1, 3")
print("->", out / "guidance.png")
