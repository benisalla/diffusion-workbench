# Evaluation

One evaluation for every method (`src/diffusion/evaluate.py`, `diffusion-eval`).

## Classifier-feature FID (not Inception FID)

FID compares the statistics of features of real and generated images: fit a Gaussian to each set of
feature vectors and compute the Fréchet distance

    FD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 (Σ_r Σ_g)^½)

(`metrics.frechet_distance`; the matrix square root uses an eigendecomposition, no scipy).
The standard FID uses Inception-v3 features of 299 x 299 ImageNet images, which makes no sense for 28
x 28 digits and needs a large download. Here the features come from **a small CNN trained on the same
dataset** (`evaluate.Classifier`: three convolutions, global average pooling, 64-d features; ~97% test
accuracy on MNIST after 1 500 steps, cached as `checkpoints/classifier_<dataset>.pt`).

What the number means and does not mean:

- lower is better; it moves when samples are blurry, noisy, of the wrong class mix, or not diverse;
- it is comparable **between runs of this repo on the same dataset with the same N**, not with any
  published FID;
- with few samples FID is biased upwards. Every table therefore gives the **"real vs real" floor**:
  N real training images scored against the test set, the best value reachable with that N.

## Label accuracy (conditional models)

For a class-conditional model, generate images for labels 0..K-1 and count how often the same
classifier predicts the requested label. With guidance `w` this measures how much the label is
obeyed; FID at the same time measures what is lost in diversity.

## NFE sweep

NFE = number of network evaluations per generated image, the real cost of sampling. Each checkpoint is
sampled with 5, 10, 20, 50 and 100 NFE for its main sampler, and at 10 and 50 for its second one.
Second-order samplers (Heun) cost two evaluations per step except the last, so they hit odd budgets
(`2N - 1`: 5, 9, 19, 49, 99); the NFE is counted by wrapping the network (`sample.NFECounter`), not
assumed. With guidance, one "evaluation" runs the network on a doubled batch.

```bash
diffusion-eval --train-classifier mnist                       # once (also done automatically)
diffusion-eval runs/edm/last.pt --nfe 5 10 20 50 100 --n-samples 1000
diffusion-eval runs/flow-matching-cfg/last.pt --guidance 0 1 3 --nfe 20
```

Writes `eval.json` (one row per sampler / NFE / guidance: FID, label accuracy, seconds per 100 images)
and `nfe.png` next to the checkpoint.

## The runs behind the README

```bash
diffusion-data mnist
diffusion-eval --train-classifier mnist            # 1 500 steps, 96.5% test accuracy, cached
python scripts/compare_all.py                      # 9 runs: train 500 steps + NFE sweep + figures, runs/<name>/
python scripts/results_table.py                    # assets/results.{json,md}, nfe_curves.png, samples_all.png, ...
```

Each run of `scripts/compare_all.py` (`--only <name>` for one) trains with the same data, network size,
steps and seed, then evaluates 200 samples per point: the main sampler at NFE 5 / 10 / 20 / 50 / 100,
the second sampler at NFE 10 / 50, and for the conditional run guidance w = 0 and 3 at NFE 20. The table
and the reading of it are in the README; the raw rows (FID, label accuracy, seconds per 100 images) are
in `assets/results.json`.

How to read the FID values: the floor is 0.13 (200 real images); 1-3 is "digits, a bit wobbly"; 10-20
is "the right strokes but blocky or noisy"; above 100 a large part of the samples is still noise
(see `assets/samples_all.png`). With 200 samples the estimate is noisy: the floor itself ranges from 0.12 to 0.18 over six random
draws of 200 real images (mean 0.14, std 0.024), and the spread grows with the FID value, so
differences of a few tenths between runs around FID 1-3 should not be over-read.
