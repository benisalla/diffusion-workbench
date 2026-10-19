# legacy/

The Centrale coursework notebook this project started from, kept verbatim for reference. The
maintained code is the PyTorch package in [`src/diffusion`](../src/diffusion).

| file | what it did |
|---|---|
| `BE2_PART_004_DIFUSION.ipynb` | text-to-image *inference* with pretrained models through Hugging Face `diffusers`: Stable Diffusion 3.5 Medium with its transformer quantised to 4-bit NF4 (`bitsandbytes`), CPU offload and attention slicing, one prompt at 256 x 256; then the much lighter `OFA-Sys/small-stable-diffusion-v0` on another prompt |

The notebook used diffusion models as a black box: a pipeline call with `num_inference_steps=30`
and `guidance_scale=4.5`. Those two numbers are exactly what this repository opens up: what a
"step" is (one network evaluation of a reverse-process solver: DDPM, DDIM, Euler, Heun, ...), why
30 is enough for some formulations and not others (the NFE curves in the README), and what
classifier-free guidance does (`methods.guided`, example 04). SD3 itself is a rectified-flow
(flow-matching) model with a transformer backbone: the `flow-matching` method with `--backbone dit`
is the same idea at 28 x 28 pixels.

## What changed, and why

Problems found in the notebook, and how the package handles them:

| Notebook | Effect | Now |
|---|---|---|
| everything runs through `diffusers` pipelines and pretrained weights | nothing about the forward process, the loss, the sampler or the guidance is visible or changeable | every formulation is ~60 readable lines in `methods.py`, trained from random weights by one loop in `train.py` |
| `!huggingface-cli login` in a cell, SD 3.5 is a gated model | the notebook stops on an interactive prompt and fails without a token and an accepted licence | no external weights at all; datasets are torchvision downloads or generated offline (`shapes`) |
| `bitsandbytes` 4-bit quantisation and `torch.float16` | needs a CUDA GPU: the notebook cannot run on a CPU | small models (0.2-1.3 M parameters) that train on a CPU in minutes; the notebook uses a GPU only to go bigger |
| the two `!pip install` lines install `diffusers transformers accelerate safetensors` twice; `torch` and `StableDiffusion3Pipeline` imported twice; `torch.cuda.empty_cache()` before anything was loaded | noise, no effect | `pip install -e .` with four dependencies (torch, torchvision, numpy, matplotlib) |
| `height=256, width=256` for a model trained around 1024 px | blurry, badly composed images (the model is far outside its training resolution) | models are trained and sampled at the resolution of their data (28 x 28 MNIST / FashionMNIST, 32 x 32 CIFAR-10 in the notebook) |
| `enable_attention_slicing()` on a PyTorch 2 pipeline | slicing trades speed for memory; with PyTorch 2's fused attention it mostly makes sampling slower | the backbones use `F.scaled_dot_product_attention` directly |
| no `generator` / seed in either pipeline call | the images cannot be reproduced | every sampler takes a `torch.Generator`; `--seed` on every CLI |
| `num_inference_steps=30`, `guidance_scale=4.5` fixed, one image per prompt | no way to see the quality-vs-steps or guidance trade-offs | `diffusion-eval` sweeps NFE 5-100 for every sampler and scores the samples; guidance is a flag (`--guidance`) with a measured label-accuracy / FID trade-off |
| images only saved to disk (`generated_image_1.png`) | no output visible in the notebook, nothing measured | grids, denoising trajectories, loss curves and a classifier-feature FID, all produced by the package |
| no tests | — | 45 pytest checks, including exact-denoiser checks of every sampler and an overfit test |
