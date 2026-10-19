# Training

One loop trains every method and backbone (`src/diffusion/train.py`):

```python
for step in range(max_steps):
    x0, y = random batch                       # tensors already in memory, values in [-1, 1]
    y[rand < p_uncond] = null                  # only with --cond: label dropout for classifier-free guidance
    loss = method.loss(net, x0, y)             # the ONLY line that depends on the method
    loss.backward(); clip_grad_norm(1.0); AdamW step
    ema = lerp(ema, net, 1 - decay)            # exponential moving average of the weights
```

Why each piece is there:

- **`method.loss`** draws a random noise level per image, builds the noisy input and returns the
  regression loss of that formulation (see [architecture.md](architecture.md)). Each image of a batch
  gets a different noise level, so one batch trains the whole range of levels at once.
- **AdamW, learning-rate warm-up (100 steps), gradient clipping at 1.0**: a diffusion loss is noisy
  (random `t`, random `ε`), and the first steps of a freshly initialised network can produce large
  gradients; warm-up and clipping keep the high learning rate (1e-3 for the small U-Net, 5e-4 for
  the DiT) from blowing up.
- **EMA of the weights** (decay 0.999, warmed up as `min(0.999, (1+step)/(10+step))` so short runs
  are not dominated by the random initial weights): samples come from the EMA copy. The raw weights
  jitter from batch to batch; a sampler chains 5-1000 network calls, and that jitter accumulates.
  The EMA is one of the cheapest, most visible quality gains in diffusion training.
- **No data loader**: MNIST is 47 MB as bytes, 188 MB as float; a batch is `x[randint(...)]`. On a
  CPU this is much faster than a `DataLoader` with workers.

## Command line

```bash
diffusion-train --method ddpm --prediction eps --schedule cosine --dataset mnist \
                --max-steps 500 --output-dir runs/ddpm-eps
diffusion-train --method flow-matching --backbone dit --lr 5e-4 --output-dir runs/flow-dit
diffusion-train --method edm --cond --output-dir runs/edm-cond           # class-conditional, for --guidance
```

| flag | default | meaning |
|---|---|---|
| `--method` | `ddpm` | `ddpm`, `score-sde`, `flow-matching`, `edm` |
| `--prediction` | `eps` | DDPM target: `eps`, `x0`, `v` |
| `--schedule` | `linear` | DDPM noise schedule: `linear`, `cosine` |
| `--sde` | `vp` | score SDE: `vp` or `ve` |
| `--backbone` | `unet` | `unet` or `dit` |
| `--base-channels`, `--channel-mults` | 16, `1 2 2` | U-Net width (32 or 64 on a GPU) |
| `--dit-dim`, `--dit-depth`, `--dit-heads`, `--patch-size` | 128, 4, 4, 4 | DiT size |
| `--dataset` | `mnist` | `mnist`, `fashion-mnist`, `cifar10`, `shapes` |
| `--cond`, `--p-uncond` | off, 0.1 | class-conditional training with label dropout |
| `--batch-size`, `--lr`, `--max-steps`, `--max-minutes` | 64, 1e-3, 2000, 0 | budget; `--max-minutes` stops early on a clock |
| `--ema-decay` | 0.999 | |
| `--save-every` | 0 | also save EMA snapshots `ckpt_<step>.pt` every N steps (used by `scripts/make_gifs.py`) |
| `--threads` | 1 | CPU threads (`torch.set_num_threads`) |
| `--output-dir` | `checkpoints/run` | |

Outputs in `--output-dir`: `last.pt` (configs + EMA weights + raw weights + history), `history.json`
(loss every `--log-every` steps, wall time, parameter count), `loss.png`, `samples.png` (64 samples
from the EMA weights, 50 sampler steps).

## Sizes and time

| setting | U-Net 16 ch (default) | U-Net 32 ch | DiT 128 x 4 |
|---|---|---|---|
| parameters (MNIST) | 0.21 M | 0.83 M | 1.25 M |
| CPU, 1 thread, batch 64 (shared, busy machine, measured) | 0.4-1 s / step | ~1.9 s / step | 0.5-1 s / step |

The README results use 500 steps of batch 64 (about half an MNIST epoch): enough to see every method
learn and to rank them, not enough for clean digits. The notebook trains for tens of thousands of
steps on a GPU.

## Scripts

`scripts/compare_all.py` trains and evaluates every README run (`--only <name>` for one), with the
same data, network, steps and evaluation; `scripts/results_table.py` gathers them into `assets/`.
