# Architecture: one network, four formulations, many samplers

Everything in this repository is built from three pieces that plug into each other:

```
data (N, C, H, W) in [-1, 1]
   │
   ▼
method.loss(net, x0, y)      ← methods.py: DDPM | ScoreSDE (VP / VE) | FlowMatching | EDM
   │   picks a noise level, makes x_t, asks the network for a target
   ▼
net(x, t, y) -> (B, C, H, W) ← backbones.py: UNet | DiT (same call for every method)
   │
   ▼
method.sample(net, n, steps, sampler, y, guidance)   ← the reverse process: DDPM, DDIM, Euler-Maruyama,
                                                        probability-flow ODE, Euler, Heun
```

The network never knows which method it serves. It receives an image and a number (a time or noise
level mapped to about [0, 1000]) and returns an image-shaped tensor. **What that tensor means — the
noise, the clean image, a velocity — is decided by the method.** That is what makes the comparison
fair: the same weights budget, the same architecture, only the formulation changes.

## Why all these methods are "the same thing"

All of them share one idea: corrupt data with Gaussian noise at many levels, train a network to undo
a little of it, then start from pure noise and undo it step by step. They differ in three choices:

| choice | DDPM | score SDE | flow matching | EDM |
|---|---|---|---|---|
| noisy sample | `sqrt(ā_t) x0 + sqrt(1-ā_t) ε` | `m(t) x0 + s(t) ε` | `(1-t) x0 + t ε` | `x0 + σ ε` |
| time / level | discrete `t ∈ {0..999}` | continuous `t ∈ [0.001, 1]` | `t ∈ [0, 1]` | `σ ∈ [0.002, 80]` |
| network target | `ε`, `x0` or `v` | `ε` (score = `-ε / s`) | velocity `ε - x0` | preconditioned `F` |
| sampler | DDPM ancestral, DDIM | Euler-Maruyama (SDE), Heun (ODE) | Euler, Heun | Heun, Euler |

Given any one of `ε̂`, `x̂0`, `v̂` or the score at a known noise level, the others follow by algebra.
So the targets carry the same information; they differ in **how hard they are to learn** at each
noise level and in **how well a sampler can use them with few steps**. The NFE curves in the README
measure exactly that.

## DDPM (`--method ddpm`)

*Ho, Jain & Abbeel 2020.* Code: `methods.DDPM`.

**Forward process.** A fixed Markov chain adds a little noise at each of `T = 1000` steps,
`q(x_t | x_{t-1}) = N(sqrt(1-β_t) x_{t-1}, β_t I)`. Composing Gaussians gives the closed form we
actually use, with `ā_t = Π_{s≤t} (1-β_s)`:

    x_t = sqrt(ā_t) x0 + sqrt(1 - ā_t) ε,     ε ~ N(0, I)

so a training example at any `t` costs one line, no loop over steps (`DDPM.q_sample`).

**Schedules** (`--schedule`). *linear*: `β` from 1e-4 to 0.02 (the paper). *cosine* (Nichol &
Dhariwal 2021): `ā_t = cos²((t/T + 0.008)/1.008 · π/2)`. Why it matters: the linear schedule
destroys a 28 px image long before `t = T` (example 01 prints the signal-to-noise ratio
`ā/(1-ā)`: 0.08 at t = 500 for linear, 0.97 for cosine), so many steps are spent on near-pure noise.
The cosine schedule spreads the useful noise levels evenly.

**Objective.** The reverse model `p(x_{t-1} | x_t)` is Gaussian; the ELBO is a sum of KL divergences
between Gaussians, and with the mean written as a function of a noise prediction it reduces, up to
per-`t` weights that the paper drops, to

    L_simple = E_{t, x0, ε} || ε - ε_θ(x_t, t) ||²

i.e. *predict the noise that was added*. Dropping the weights over-weights the hard, noisy levels,
which helps sample quality.

**Parameterisations** (`--prediction`). The network can predict
- `eps`: the noise (the paper),
- `x0`: the clean image directly,
- `v = sqrt(ā) ε - sqrt(1-ā) x0` (Salimans & Ho 2022): a mix that is well scaled at *every* noise
  level. `ε` prediction becomes useless near pure noise (x0 = (x_t - sqrt(1-ā) ε)/sqrt(ā) divides by a
  tiny number) and `x0` prediction is useless near clean data; `v` interpolates between the two.

`DDPM.to_x0_eps` converts any of the three into `(x̂0, ε̂)` and the samplers only use those, so the
three are trained differently but sampled identically. `x̂0` is clipped to [-1, 1] (the data range).

**Samplers.** One update covers both (Song, Meng & Ermon 2021, DDIM eq. 12): going from `t` to an
earlier `s` on a strided subset of the 1000 steps,

    x_s = sqrt(ā_s) x̂0 + sqrt(1 - ā_s - σ²) ε̂ + σ z,   σ = η sqrt((1-ā_s)/(1-ā_t)) sqrt(1 - ā_t/ā_s)

- `--sampler ddpm` (η = 1): the ancestral sampler; with all 1000 steps it is exactly DDPM's
  `p(x_{t-1} | x_t)` with variance `β̃_t`. Fresh noise at every step.
- `--sampler ddim` (η = 0): deterministic. The same trained model, no noise injected: the update
  follows an ODE (it is Euler's method on the probability-flow ODE in disguise), which is why it
  stays good with 10-50 steps while the ancestral sampler needs hundreds.

## Score-based SDEs (`--method score-sde --sde vp|ve`)

*Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021.* Code: `methods.ScoreSDE`.

Time becomes continuous and the forward process an SDE `dx = f(x, t) dt + g(t) dw` whose marginal is
`x_t = m(t) x0 + s(t) ε`:

| | `f(x, t)` | `g(t)²` | `m(t)`, `s(t)` |
|---|---|---|---|
| VP (variance preserving, the continuous DDPM) | `-β(t) x / 2` | `β(t)`, linear 0.1 → 20 | `m = exp(-¼ t²(β_max-β_min) - ½ t β_min)`, `s = sqrt(1 - m²)` |
| VE (variance exploding, NCSN / SMLD) | `0` | `d s²/dt = 2 s² ln(σ_max/σ_min)` | `m = 1`, `s = 0.01 · 5000^t` |

**What is learned: the score** `∇_x log p_t(x)`. For the Gaussian `q(x_t | x0)` it is
`-(x_t - m x0)/s² = -ε / s`, so denoising score matching is again ε-prediction:
`score_θ(x, t) = -ε_θ(x, t) / s(t)`, loss `|| ε_θ - ε ||²` (the `λ(t) = s²` weighting).
For VE the noisy input reaches a standard deviation of 50, so the network sees `x / sqrt(m² + s²)`
(unit scale), the same trick EDM formalises.

**Reverse time** (Anderson 1982): the SDE `dx = [f - g² ∇log p_t] dt + g dw̄` run from `t = 1` to 0
produces data. Two samplers:

- `--sampler em`: Euler-Maruyama on that reverse SDE (drift step + fresh noise each step).
- `--sampler ode`: Heun's method on the **probability-flow ODE** `dx = [f - ½ g² ∇log p_t] dt`, which
  has the same marginals `p_t` and no noise at all. Deterministic, so it tolerates large steps.

Both end with one denoising step `x̂0 = (x - s ε̂)/m` (Song et al.'s "noise removal"). Without it the last
Euler step overshoots: near `t = 0` the step `g² Δt / s²` is larger than 1. (The exact-denoiser test
in `tests/test_methods.py` caught this.)

## Flow matching / rectified flow (`--method flow-matching`)

*Lipman et al. 2023; Liu, Gong & Liu 2023 (rectified flow); also the formulation of Stable Diffusion 3.*
Code: `methods.FlowMatching`.

Forget SDEs and schedules: draw a straight line from a data point to a noise sample,

    x_t = (1 - t) x0 + t ε,    t ∈ [0, 1]

Its time derivative is the constant velocity `ε - x0`, and the network learns to predict it:
`L = || v_θ(x_t, t) - (ε - x0) ||²`. The regression target is the *conditional* velocity of one pair;
its minimiser is the average velocity of all lines passing through `x_t`, which defines an ODE
`dx/dt = v_θ(x, t)` that transports the noise distribution to the data distribution.
Sampling integrates it from `t = 1` to 0 with `--sampler euler` or `heun`.

Why it is popular: the simplest loss of the four, no noise schedule to tune, and paths that are
nearly straight, so few Euler steps already work. `x̂0 = x - t v̂` (the line extended to `t = 0`) is
what the trajectory figures show in their second row.

## EDM (`--method edm`)

*Karras, Aittala, Aila & Laine 2022, "Elucidating the Design Space of Diffusion-Based Generative
Models".* Code: `methods.EDM`.

EDM writes the noisy sample as `x = x0 + σ ε` and asks: which network input, output and loss
weighting make training easiest at *every* `σ` from 0.002 to 80? The answer is a preconditioned denoiser

    D(x, σ) = c_skip(σ) x + c_out(σ) F_θ(c_in(σ) x, c_noise(σ))
    c_in = 1/sqrt(σ² + σ_d²)   c_skip = σ_d²/(σ² + σ_d²)   c_out = σ σ_d / sqrt(σ² + σ_d²)   c_noise = ¼ ln σ

with `σ_d = 0.5` the data standard deviation. In plain words: the network input always has unit
variance (`c_in`); at low noise `D ≈ x` and the network only predicts a small correction, at high
noise `D ≈ c_out F` and the network predicts the clean image from scratch; and the regression target
of `F` has unit variance at every `σ`. Training draws `ln σ ~ N(-1.2, 1.2²)`, concentrating effort on
the middle noise levels where the loss can actually decrease.

Sampling solves `dx/dσ = (x - D(x, σ))/σ` with Heun's method on the schedule
`σ_i = (σ_max^{1/ρ} + i/(N-1)(σ_min^{1/ρ} - σ_max^{1/ρ}))^ρ`, `ρ = 7`, which takes small steps near
the data where details are decided. A Heun step costs two network calls, except the last one
(to `σ = 0`), so `N` steps = `2N - 1` NFE. The stochastic "churn" of the paper is not implemented: the
deterministic sampler is the one the paper recommends for this kind of model.

## Classifier-free guidance (`--cond`, `--guidance w`)

*Ho & Salimans 2022.* Code: `methods.guided`, label dropout in `train.train`.

Training: the class label is fed to the network (an embedding added to the time embedding), but 10%
of the time it is replaced by a "null" label (index `num_classes`). One network thus learns both the
conditional and the unconditional model. Sampling: at every network call,

    out = out(x, t, null) + w · (out(x, t, y) - out(x, t, null))

`w = 1` is the plain conditional model, `w = 0` ignores the label, `w > 1` extrapolates away from the
unconditional prediction: samples become more typical of their class and less diverse. Because the
combination is linear and every target in this repo (ε, v, velocity, EDM's F) is linear in the
score, the same function guides all four methods. One guided call runs the network on a doubled batch.

## Backbones (`--backbone unet|dit`)

Both map `(x: (B, C, H, W), t: (B,), y: (B,) or None) -> (B, C, H, W)`. The time `t` goes through
sinusoidal features and an MLP; a label embedding is added to it. The last layer is zero-initialised,
so every network starts by predicting 0 (a stable start for every target).

**U-Net** (`backbones.UNet`, default 16 channels, 0.21 M parameters at 28 px):

| stage | resolution | channels | blocks |
|---|---|---|---|
| input conv | 28 x 28 | 16 | conv 3x3 |
| down 1 | 28 x 28 | 16 | ResBlock, stride-2 conv |
| down 2 | 14 x 14 | 32 | ResBlock, stride-2 conv |
| down 3 | 7 x 7 | 32 | ResBlock + self-attention |
| middle | 7 x 7 | 32 | ResBlock, attention, ResBlock |
| up 3, 2, 1 | 7 → 14 → 28 | 32, 32, 16 | concat skip, ResBlock (+ attention at 7 x 7), nearest upsample + conv |
| output | 28 x 28 | C | GroupNorm, SiLU, conv 3x3 (zero init) |

A ResBlock is GroupNorm-SiLU-conv, plus the conditioning vector projected to the channels and added,
then GroupNorm-SiLU-conv, with a residual connection. Convolutions give the U-Net a strong locality
prior, which is why it learns fast on small images.

**DiT** (`backbones.DiT`, Peebles & Xie 2023, 1.25 M parameters with the defaults): cut the image into
4 x 4 patches (28 px → 7 x 7 = 49 tokens of dimension 128), add a learned position embedding, run 4
Transformer blocks, project each token back to a 4 x 4 patch. The conditioning enters through
**adaLN-Zero**: it produces a shift, a scale and a gate for each LayerNorm / branch, and the gates
start at zero so each block starts as the identity. No convolutional prior: it needs more steps than
the U-Net on small data, but it scales better, which is why large models (SD3, DiT-XL) use it.

## Files

| file | contents |
|---|---|
| `config.py` | `ModelConfig`, `MethodConfig`, `TrainConfig`, `RunConfig` (saved in every checkpoint) |
| `backbones.py` | time / label conditioning, `UNet`, `DiT`, `build_backbone` |
| `methods.py` | `guided`, `DDPM`, `ScoreSDE`, `FlowMatching`, `EDM`, `build_method` |
| `data.py` | MNIST / FashionMNIST / CIFAR-10 / shapes as tensors, `diffusion-data` |
| `train.py` | the loop, EMA, checkpoints, `diffusion-train` |
| `sample.py` | `load_checkpoint`, `generate`, `NFECounter`, `diffusion-sample` |
| `evaluate.py` | feature classifier, classifier-feature FID, label accuracy, NFE sweep, `diffusion-eval` |
| `metrics.py` | Fréchet distance between Gaussians |
| `viz.py` | grids, trajectories, loss and NFE curves |
