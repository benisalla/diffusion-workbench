# Data

`diffusion.data.load_dataset(name, data_dir, train)` returns `(images, labels, num_classes)` with
images as one float tensor `(N, C, H, W)` in **[-1, 1]**. Why [-1, 1]: every method assumes data
roughly centred with unit-ish scale, so that pure noise `N(0, I)` and data live on comparable
scales (EDM's `σ_data = 0.5` is close to MNIST's 0.6 standard deviation in this range), and the DDPM
sampler clips `x̂0` to this interval.

| `--dataset` | images | classes | size | how it is obtained |
|---|---|---|---|---|
| `mnist` | 60 000 train / 10 000 test, 1 x 28 x 28 | 10 digits | 11 MB | torchvision download into `data/mnist` |
| `fashion-mnist` | 60 000 / 10 000, 1 x 28 x 28 | 10 clothing types | 30 MB | torchvision download into `data/fashion-mnist` |
| `cifar10` | 50 000 / 10 000, 3 x 32 x 32 | 10 objects | 163 MB | torchvision download (meant for the GPU notebook) |
| `shapes` | 12 000 / 2 000, 1 x 28 x 28 | circle, square, triangle, cross | — | generated from a seed (`make_shapes`), offline |

```bash
diffusion-data mnist            # download once, print statistics, write data/mnist_preview.png
diffusion-data shapes           # nothing to download: preview only
```

The **shapes** set exists so that every test and example runs without a network connection: one
filled shape per image, random size, position and grey level. It is easier than MNIST (four simple
classes) and learns in a few hundred steps, which makes it a good first check of a new method.

Everything is held in memory and batches are drawn with `x[torch.randint(...)]`; with
`--n-train N` a random subset of N training images is used (fast experiments).

The DiT backbone needs the image side to be divisible by `--patch-size` (28 / 4 = 7, 32 / 4 = 8).
The U-Net halves the resolution once per entry of `--channel-mults` minus one: `1 2 2` gives
28 → 14 → 7 (and 32 → 16 → 8 for CIFAR-10).
