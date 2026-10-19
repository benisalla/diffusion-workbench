"""Build notebooks/train_colab_kaggle.ipynb: the step-by-step explanation + real GPU runs.

The notebook installs this repository from GitHub and imports the `diffusion` package, so the code
it runs is exactly the code in src/. Each method's key function is printed next to the markdown that
explains it (`show(...)` prints the package source). Regenerate after editing:

    python notebooks/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("train_colab_kaggle.ipynb")
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# diffusion-workbench: diffusion and flow models, step by step (Kaggle & Colab)

This notebook explains, then trains and compares, the image-generation formulations of the
[`diffusion-workbench`](https://github.com/benisalla/diffusion-workbench) repository, on a free GPU:

| method | what the network learns | how images are sampled |
|---|---|---|
| **DDPM** (`ddpm`, `--prediction eps / x0 / v`) | the noise added to an image (or the clean image, or `v`) | ancestral sampling, or **DDIM** (deterministic, few steps) |
| **score SDE** (`score-sde`, VP or VE) | the score ∇ log p_t(x) | reverse SDE (Euler-Maruyama) or probability-flow ODE (Heun) |
| **flow matching** (`flow-matching`) | the velocity of a straight line from data to noise | Euler or Heun on an ODE |
| **EDM** (`edm`) | a preconditioned denoiser | Heun on Karras' noise schedule |
| **classifier-free guidance** (`--cond`, `guidance=w`) | the same model with and without the label | extrapolate between the two |

All of them share **one U-Net (or one DiT), one training loop, one sampler interface and one
evaluation**, so the differences you see come from the formulation only.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Session limit | 12 h per session, 30 h GPU / week | ~12 h, may disconnect when idle |
| Outputs kept in | `/kaggle/working` (saved with "Save Version") | `/content` (download them at the end) |

**Just trying it?** Set `SMOKE_TEST = True` in section 2: tiny settings on the offline shapes data,
every cell runs in a few minutes, even on a CPU.

**Sections:** 1 Install · 2 Config · 3 Data · 4 Forward process · 5 Reverse process and the ε objective
· 6 DDIM · 7 x0 / v prediction · 8 Score matching and SDEs · 9 Flow matching · 10 EDM · 11 Classifier-free
guidance · 12 Backbones · 13 Train every method · 14 Evaluate: FID vs NFE · 15 Guidance in practice ·
16 Save
""")

md("""
## 1. Install

Kaggle and Colab already have PyTorch, torchvision, NumPy and matplotlib. This installs the
`diffusion` package itself from GitHub (the repository is small: pure Python, four dependencies).
If the repository is private for you, add a `GITHUB_TOKEN` secret (Kaggle *Add-ons → Secrets*,
Colab *🔑 Secrets*) or upload the repo folder and `pip install -e` it.
""")
code("""
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("diffusion") is None:
    token = os.environ.get("GITHUB_TOKEN", "")
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore

        token = token or userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
    auth = f"{token}@" if token else ""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           f"git+https://{auth}github.com/benisalla/diffusion-workbench.git"])

import inspect
import json
import time

import matplotlib.pyplot as plt
import torch
from IPython.display import Code, Image, display

from diffusion import backbones, methods
from diffusion.backbones import build_backbone, count_parameters
from diffusion.config import MethodConfig, RunConfig
from diffusion.data import CLASS_NAMES, load_dataset
from diffusion.evaluate import Evaluator, evaluate_checkpoint, get_classifier
from diffusion.methods import DDPM, EDM, FlowMatching, ScoreSDE, build_method, guided, method_label
from diffusion.sample import generate, load_checkpoint
from diffusion.train import train
from diffusion.viz import make_grid, plot_losses, plot_nfe_curves, save_grid, save_trajectory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")


def show(obj):
    \"\"\"Print the package's source code of a function or class, so the maths below sits next to the code.\"\"\"
    display(Code(inspect.getsource(obj), language="python"))


def grid(x, nrow=8, title=None, size=5):
    g = make_grid(x, nrow)
    plt.figure(figsize=(size, size * g.shape[0] / g.shape[1]))
    plt.imshow(g.squeeze(-1).numpy() if g.shape[-1] == 1 else g.numpy(), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
""")

md("""
## 2. Config

`DATASET = "cifar10"` (32 x 32 colour, the real run) or `"mnist"` / `"fashion-mnist"` (28 x 28).
`STEPS` training steps per method. The first run prints its speed: multiply by the number of runs
in `RUNS` (+2 for the guidance and DiT runs) to check it fits your session, and lower `STEPS` or
shorten `RUNS` if it does not.
""")
code("""
SMOKE_TEST = False          # True: tiny settings on the offline shapes data (CPU friendly)

DATASET = "cifar10"
STEPS = 10_000
BATCH = 128
BASE_CHANNELS = 64          # U-Net width (the repo's CPU default is 16)
N_EVAL = 2_000              # generated images per FID estimate
NFES = (5, 10, 20, 50, 100)
OUT = "/kaggle/working/runs" if os.path.isdir("/kaggle/working") else "runs"

# name -> (method config, extra settings)
RUNS = {
    "ddpm-eps": MethodConfig("ddpm", "eps", "cosine"),
    "ddpm-v": MethodConfig("ddpm", "v", "cosine"),
    "score-sde-vp": MethodConfig("score-sde", sde="vp"),
    "flow-matching": MethodConfig("flow-matching"),
    "edm": MethodConfig("edm"),
}
CFG_RUN = True               # also train a class-conditional flow-matching model for guidance
DIT_RUN = True               # also train flow matching with the DiT backbone

if SMOKE_TEST:
    DATASET, STEPS, BATCH, BASE_CHANNELS, N_EVAL, NFES = "shapes", 60, 32, 16, 100, (5, 10, 20)
os.makedirs(OUT, exist_ok=True)
""")

md("""
## 3. Data

Images are loaded once into one tensor in **[-1, 1]**: the methods assume data roughly centred with
unit-ish scale, so that pure Gaussian noise and data live on comparable scales.
""")
code("""
x_train, y_train, NUM_CLASSES = load_dataset(DATASET, "data", train=True)
print(DATASET, tuple(x_train.shape), f"{NUM_CLASSES} classes", "std", round(x_train.std().item(), 3))
grid(x_train[:64], title=f"{DATASET}: real images")
""")

md(r"""
## 4. The forward process: destroying an image on purpose

A diffusion model is defined by how it turns data into noise. DDPM uses a Markov chain of `T = 1000`
small Gaussian steps `q(x_t | x_{t-1}) = N(sqrt(1-β_t) x_{t-1}, β_t I)`. Composing Gaussians gives a
closed form for any `t` directly, with `ā_t = Π (1 - β_s)`:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)$$

**Why it matters:** we never simulate the chain during training; one random `t` per image and one line
of code give a training example. The schedule `β_t` decides how fast information disappears. The
signal-to-noise ratio `ā/(1-ā)` shows the difference between the linear schedule of the DDPM paper
and the cosine schedule of Nichol & Dhariwal: the linear one has already destroyed most of a small
image halfway through, so many steps are wasted on near-pure noise.
""")
code("show(DDPM.q_sample)")
code("""
x0 = x_train[:1].repeat(6, 1, 1, 1)
eps = torch.randn_like(x0)
rows = []
for schedule in ("linear", "cosine"):
    m = DDPM(1000, schedule)
    t = torch.tensor([0, 100, 250, 500, 750, 999])
    rows.append(m.q_sample(x0, t, eps))
    snr = (m.abar / (1 - m.abar))[t]
    print(f"{schedule:6s} SNR at t = {t.tolist()}: " + ", ".join(f"{v:.3g}" for v in snr.tolist()))
grid(torch.cat(rows), nrow=6, title="linear (top) vs cosine (bottom), t = 0 ... 999")
""")

md(r"""
## 5. The reverse process and the ε objective

To generate, we run the chain backwards: `p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t² I)`. Training
maximises a variational lower bound (ELBO) of `log p_θ(x_0)`. The ELBO is a sum, over `t`, of KL
divergences between the true posterior `q(x_{t-1} | x_t, x_0)` — a Gaussian we can write down because
we know `x_0` — and the model's Gaussian. A KL between two Gaussians with the same variance is a
squared distance between means. Writing the model's mean in terms of a **noise prediction** `ε_θ`,

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \varepsilon_\theta(x_t, t)\Big)$$

each KL term becomes a weighted `||ε - ε_θ(x_t, t)||²`. Ho et al. drop the weights:

$$L_{\text{simple}} = \mathbb E_{t, x_0, \varepsilon}\, \| \varepsilon - \varepsilon_\theta(x_t, t)\|^2$$

**In plain words:** show the network a noisy image and the noise level; ask it which noise was added.
Dropping the weights puts more emphasis on the noisy levels, which are harder and matter more for
sample quality. Here is the whole loss in the package:
""")
code("show(DDPM.loss)")

md(r"""
## 6. DDIM: the same model, a deterministic sampler with few steps

The ancestral sampler adds fresh noise at every one of the 1000 steps. Song, Meng & Ermon (2021)
noticed that the DDPM *training* objective only depends on the marginals `q(x_t | x_0)`, so many
reverse processes are compatible with the same trained network. From `x_t`, predict `x̂_0` and `ε̂`, and
jump to any earlier `s`:

$$x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\,\hat\varepsilon + \sigma z,\qquad
\sigma = \eta \sqrt{\tfrac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_s}}$$

`η = 1` is DDPM's ancestral sampler; `η = 0` is **DDIM**: no noise, a deterministic map from the
initial noise to the image (an ODE solver in disguise), which keeps working with 10-50 steps.
""")
code("show(DDPM.sample)")

md(r"""
## 7. What should the network predict? ε, x0 or v

`ε`, `x_0` and `v = sqrt(ā) ε - sqrt(1-ā) x_0` (Salimans & Ho 2022) carry the same information:
given `x_t` and `t`, each one determines the other two. But they are not equally easy to learn:

- `ε`-prediction is great at low noise, but near pure noise `x̂_0 = (x_t - sqrt(1-ā) ε̂) / sqrt(ā)`
  divides by a tiny number: small errors in `ε̂` explode.
- `x_0`-prediction is the opposite: at low noise the network has to copy its input almost exactly.
- `v` is well scaled everywhere (it rotates between `ε` and `-x_0` as the noise grows), which is why
  it is the default of many modern models and the only sensible choice with a schedule that reaches
  pure noise.

`to_x0_eps` converts the network output, whatever it is, back to `(x̂_0, ε̂)`:
""")
code("show(DDPM.to_x0_eps)")

md(r"""
## 8. Score matching and SDEs

Song et al. (2021) let `t` become continuous: the forward process is an SDE
`dx = f(x, t) dt + g(t) dw` with Gaussian marginals `x_t = m(t) x_0 + s(t) ε`.

- **VP** (variance preserving): `f = -½ β(t) x`, `g² = β(t)` — the continuous limit of DDPM.
- **VE** (variance exploding): `f = 0`, the noise `s(t)` grows geometrically from 0.01 to 50 — the
  continuous limit of noise-conditional score networks (NCSN).

The network learns the **score** `∇_x log p_t(x)`, the direction in which the noisy density increases.
Denoising score matching shows that for the Gaussian `q(x_t | x_0)` the target is `-ε / s(t)`, so it is
again ε-prediction, rescaled. Anderson's theorem gives the **reverse-time SDE**
`dx = [f - g² ∇log p_t(x)] dt + g dw̄`; there is also a **probability-flow ODE**
`dx = [f - ½ g² ∇log p_t(x)] dt` with the same marginals and no randomness. The package integrates the
first with Euler-Maruyama (`em`) and the second with Heun (`ode`):
""")
code("show(ScoreSDE._velocity)")
code("show(ScoreSDE.sample)")

md(r"""
## 9. Flow matching: straight lines from data to noise

Flow matching (Lipman et al. 2023) / rectified flow (Liu et al. 2023) drops schedules and SDEs. Draw a
straight line between a data point and a noise sample, `x_t = (1-t) x_0 + t ε`; its velocity is the
constant `ε - x_0`; regress it:

$$L = \mathbb E\, \| v_\theta(x_t, t) - (\varepsilon - x_0) \|^2$$

The minimiser is the *average* velocity of all lines through `x_t`, a vector field whose ODE
`dx/dt = v_θ` carries the noise distribution onto the data. Paths are close to straight, so a few Euler
steps already give images. This is the formulation of Stable Diffusion 3.
""")
code("show(FlowMatching)")

md(r"""
## 10. EDM: designing the network's input and output

Karras et al. (2022) write `x = x_0 + σ ε` and wrap the network `F_θ` into a denoiser

$$D(x, \sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x,\ c_{\text{noise}}(\sigma)\big)$$

chosen so that the network's input and its regression target both have unit variance at **every** σ
from 0.002 to 80: at low noise `D ≈ x` and `F` predicts a small correction; at high noise `F` predicts
the image. Training samples `ln σ ~ N(-1.2, 1.2²)`. Sampling uses Heun's second-order method on a
schedule that spends its steps near the data, where details are decided.
""")
code("show(EDM.precond)")
code("show(EDM.sample)")

md(r"""
## 11. Classifier-free guidance

Train with the class label, but replace it by a "null" label 10% of the time: the same network learns
`p(x | y)` and `p(x)`. At sampling time (Ho & Salimans 2022):

$$\text{out} = \text{out}(x, t, \varnothing) + w\,\big(\text{out}(x, t, y) - \text{out}(x, t, \varnothing)\big)$$

`w = 1` is the conditional model, `w > 1` pushes samples towards what makes the label likely: crisper,
more recognisable, less diverse. The combination is linear, and every target in this repo is linear in
the score, so the same function guides all methods:
""")
code("show(guided)")

md("""
## 12. Backbones: U-Net and DiT

Every method calls the network the same way, `net(x, t, y)`, and gets an image-shaped tensor back.
The **U-Net** (convolutions, skip connections, attention at the lowest resolution) has a strong
locality prior and learns fast on small images. The **DiT** cuts the image into patches and runs
Transformer blocks whose LayerNorms are modulated by the time / label embedding (adaLN-Zero); no
convolutional prior, slower to start, but it scales — it is what SD3 and other large models use.
""")
code("""
xb = x_train[:2]
for bb in ("unet", "dit"):
    cfg = RunConfig().model
    cfg.backbone, cfg.in_channels, cfg.image_size = bb, xb.shape[1], xb.shape[2]
    cfg.base_channels = BASE_CHANNELS
    net = build_backbone(cfg)
    print(f"{bb}: {count_parameters(net) / 1e6:.2f} M parameters, output {tuple(net(xb, torch.zeros(2)).shape)}")
""")

md("""
## 13. Train every method (same network, same data, same budget)

One loop for all (`diffusion.train.train`): random batch → `method.loss` → AdamW step with gradient
clipping → EMA of the weights. Only the `method.loss` line differs between the runs.
""")
code("show(methods.FlowMatching.loss)")
code("""
def make_cfg(name, mc, cond=False, backbone="unet"):
    cfg = RunConfig(method=mc)
    cfg.model.backbone, cfg.model.base_channels = backbone, BASE_CHANNELS
    if DATASET == "cifar10":
        cfg.model.channel_mults = (1, 2, 2)
    t = cfg.train
    t.dataset, t.max_steps, t.batch_size, t.cond = DATASET, STEPS, BATCH, cond
    t.lr = 5e-4 if backbone == "dit" else 2e-4
    t.log_every, t.output_dir, t.device, t.threads = max(1, STEPS // 50), f"{OUT}/{name}", DEVICE, 4
    return cfg


plan = dict(RUNS)
if CFG_RUN:
    plan["flow-matching-cfg"] = MethodConfig("flow-matching")
if DIT_RUN:
    plan["flow-matching-dit"] = MethodConfig("flow-matching")
results = {}
for name, mc in plan.items():
    cfg = make_cfg(name, mc, cond=name.endswith("cfg"), backbone="dit" if name.endswith("dit") else "unet")
    res = train(cfg, data=(x_train, y_train, NUM_CLASSES))
    results[name] = res
    m = res["method"]
    labels = torch.arange(64) % NUM_CLASSES if cfg.model.num_classes else None
    grid(generate(res["ema"], m, 64, m.steps_for_nfe(m.samplers[0], 50), labels=labels),
         title=f"{name}: {m.samplers[0]}, NFE ~50")
plot_losses({k: v["history"] for k, v in results.items()}, f"{OUT}/loss_curves.png")
display(Image(f"{OUT}/loss_curves.png"))
""")

md("""
## 14. Evaluate: classifier-feature FID vs number of network evaluations

A small CNN classifier is trained once on the dataset; its penultimate features of real test images
and of generated images are fitted with Gaussians and compared with the Fréchet distance (the FID
formula). **This is not Inception FID**: numbers compare methods within this notebook only. The
"real vs real" floor is what a perfect model would score with the same number of samples.

The NFE sweep samples each model with 5 to 100 network evaluations per image, for every sampler of
its method: the quality-vs-cost trade-off that motivates DDIM, ODE samplers and EDM.
""")
code("""
ev = Evaluator(DATASET, "data", OUT)
floor = ev.real_floor(N_EVAL)
print(f"real vs real floor: {floor:.2f}")
table, curves = {}, {}
for name, res in results.items():
    rows = evaluate_checkpoint(res["path"], ev, NFES, None, N_EVAL)
    table[name] = rows
    for r in rows:
        curves.setdefault(f"{name} ({r['sampler']})", []).append((r["nfe"], r["fid"]))
json.dump(table, open(f"{OUT}/results.json", "w"), indent=1)
plot_nfe_curves(curves, f"{OUT}/nfe_curves.png", floor=floor)
display(Image(f"{OUT}/nfe_curves.png"))
""")

md("""
### Watching the reverse process

Top row of each pair: the current `x_t`; bottom row: the network's guess of the final image `x̂_0`
at that step. The guess becomes sharp long before `x_t` is clean: coarse structure is decided at
high noise, details at low noise.
""")
code("""
for name in ("ddpm-eps", "flow-matching", "edm"):
    if name in results:
        m, rec = results[name]["method"], []
        generate(results[name]["ema"], m, 6, m.steps_for_nfe(m.samplers[0], 20), record=rec)
        save_trajectory(rec, f"{OUT}/{name}_trajectory.png", title=name)
        display(Image(f"{OUT}/{name}_trajectory.png"))
""")

md("""
## 15. Classifier-free guidance in practice

Same conditional model, guidance `w` from 0 (ignore the label) to 5. The classifier measures how often
a sample generated for class `c` is recognised as `c`; FID measures realism and diversity. Expect label
accuracy to rise with `w` and FID to get worse past a point: guidance trades diversity for fidelity.
""")
code("""
if CFG_RUN:
    res = results["flow-matching-cfg"]
    labels = torch.arange(N_EVAL) % NUM_CLASSES
    for w in (0.0, 1.0, 2.0, 3.0, 5.0):
        x = generate(res["ema"], res["method"], N_EVAL, 20, "euler", labels, guidance=w)
        print(f"w = {w:g}: FID {ev.fid(x):7.2f}   label accuracy {ev.label_accuracy(x, labels):.1%}")
    grid(generate(res["ema"], res["method"], 80, 20, "euler", torch.arange(80) // 8 % NUM_CLASSES, guidance=3.0),
         title="w = 3, one row per class: " + ", ".join(CLASS_NAMES[DATASET]))
""")

md("""
## 16. Save

Checkpoints (`last.pt`, loadable with the repo's `diffusion-sample` and `diffusion-eval`), curves and
`results.json` are in `OUT`. On Kaggle, *Save Version* keeps `/kaggle/working`; on Colab, zip and download.
""")
code("""
import shutil

archive = shutil.make_archive(f"{OUT}_archive", "zip", OUT)
print("archive:", archive)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
