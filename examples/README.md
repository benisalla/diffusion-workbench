# Examples

Short scripts that use the `diffusion` package from Python, in the order one would read them. All
run on a CPU in minutes on the built-in **shapes** dataset (generated, no download), from the
repository root after `pip install -e ".[dev]"`. Outputs go to `examples/output/`.

| script | what it shows | time (CPU) | needs |
|---|---|---|---|
| [01_forward_process.py](01_forward_process.py) | how each method turns an image into noise: DDPM linear vs cosine, VP / VE SDE, flow-matching line, EDM sigmas | seconds | nothing |
| [02_train_one_method.py](02_train_one_method.py) | `train()` with one `RunConfig`, then samples and the denoising trajectory; `--method`, `--prediction`, `--sde`, `--backbone` pick the formulation | 3-5 min | nothing |
| [03_samplers_and_nfe.py](03_samplers_and_nfe.py) | one checkpoint, every sampler of its method at NFE 5 / 10 / 20 / 50, scored with the classifier-feature FID | 1-3 min | 02 |
| [04_classifier_free_guidance.py](04_classifier_free_guidance.py) | class-conditional flow matching, label dropout, guidance w = 0 / 1 / 3 and how often the shape is right | 4-6 min | nothing |
| [05_compare_all_methods.py](05_compare_all_methods.py) | six formulations, same network and budget (100 steps each), one table | ~1 min per method | nothing |

```bash
python examples/01_forward_process.py
python examples/02_train_one_method.py                      # DDPM; try --method flow-matching / edm / score-sde
python examples/03_samplers_and_nfe.py                      # DDIM vs DDPM ancestral on that checkpoint
python examples/04_classifier_free_guidance.py
python examples/05_compare_all_methods.py --steps 150
```

The same steps as CLI commands are in the main README (`diffusion-data`, `diffusion-train`,
`diffusion-sample`, `diffusion-eval`); the Kaggle/Colab notebook runs them on CIFAR-10 with a GPU.
