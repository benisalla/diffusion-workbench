# Testing

```bash
OMP_NUM_THREADS=1 pytest -q          # 45 tests, under a minute on an idle CPU
```

The tests use networks with 8 channels (a few thousand parameters) and the offline shapes data, so
they need no download and no GPU.

| file | what it checks | why |
|---|---|---|
| `test_backbones.py` | U-Net and DiT, conditional or not, return the input shape; the zero-initialised output is exactly 0; the label changes the output after one training step; default sizes stay small; a bad patch size is rejected | the network contract every method relies on |
| `test_methods.py` — losses | every formulation (DDPM eps / x0 / v, VP, VE, flow matching, EDM) gives a finite loss and gradients for every parameter | catches shape and broadcasting errors |
| `test_methods.py` — NFE | every sampler of every method costs exactly the number of network calls `steps_for_nfe` promises (Heun: `2N - 1`) | the NFE axis of the README curves is measured, not assumed |
| `test_methods.py` — maths | DDPM schedules decrease; the three DDPM parameterisations convert back to the true `x0` and `ε`; VP marginals satisfy `m² + s² = 1`; the VE `g²` is the derivative of `s²`; EDM's `c_in` gives unit variance and its schedule hits 80 and 0.002 | the equations of [architecture.md](architecture.md) |
| `test_methods.py` — **exact denoiser** | for data made of a single image `x*`, the optimal network output is known in closed form for every method; plugging that "oracle" into every sampler (30 steps) must land on `x*` | tests the samplers independently of training: a sign error or a wrong time mapping fails here. It found that the score-SDE samplers needed a final denoising step |
| `test_methods.py` — guidance | `w = 1` gives the conditional output, `w = 0` the unconditional one, `w = 3` the extrapolation | |
| `test_metrics_data.py` | Fréchet distance of known Gaussians (0, a mean shift of 12, a 1-d variance case); same-distribution features score near 0; shapes are deterministic and in range | the evaluation number itself |
| `test_train_and_cli.py` | **overfitting a batch**: flow matching on 8 shape images must cut its loss by more than half in 80 steps; checkpoint round trip (reloaded model gives identical samples, conditional, with guidance); `diffusion-train` → `diffusion-sample` (with trajectory) → NFE evaluation end to end; the DiT trains | the whole pipeline works |
