"""Datasets as in-memory tensors in [-1, 1]: MNIST, FashionMNIST, CIFAR-10 and an offline shapes set.

The small image sets fit in RAM (MNIST: 60 000 x 28 x 28 bytes = 47 MB), so batches are drawn by
indexing one tensor: no DataLoader, no worker processes, fast on a CPU.

`diffusion-data <name>` downloads a dataset into `data/` (once) and writes a preview grid.
`shapes` needs no download: 28 x 28 images of four filled shapes (circle, square, triangle, cross)
at random positions and sizes, generated from a seed. It is there so everything runs offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DATASETS = ("mnist", "fashion-mnist", "cifar10", "shapes")
SHAPE_NAMES = ("circle", "square", "triangle", "cross")
CLASS_NAMES = {
    "mnist": tuple(str(i) for i in range(10)),
    "fashion-mnist": ("t-shirt", "trouser", "pullover", "dress", "coat",
                      "sandal", "shirt", "sneaker", "bag", "boot"),
    "cifar10": ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
    "shapes": SHAPE_NAMES,
}


def make_shapes(n: int, size: int = 28, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n grayscale images (uint8, n x size x size) of one filled shape each, and their class ids."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    images = np.zeros((n, size, size), dtype=np.uint8)
    labels = rng.integers(0, len(SHAPE_NAMES), n)
    for i, k in enumerate(labels):
        r = rng.uniform(0.18, 0.32) * size                      # half-size of the shape
        cx, cy = rng.uniform(r + 1, size - r - 1, 2)
        dx, dy = xx - cx, yy - cy
        if k == 0:
            mask = dx ** 2 + dy ** 2 <= r ** 2
        elif k == 1:
            mask = (np.abs(dx) <= 0.8 * r) & (np.abs(dy) <= 0.8 * r)
        elif k == 2:  # upward triangle: below the two slanted edges and above the base
            mask = (dy <= 0.8 * r) & (dy >= -r + 2 * np.abs(dx))
        else:
            w = 0.3 * r
            mask = ((np.abs(dx) <= w) & (np.abs(dy) <= r)) | ((np.abs(dy) <= w) & (np.abs(dx) <= r))
        images[i][mask] = int(rng.uniform(170, 255))
    return images, labels.astype(np.int64)


def _to_tensor(images: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 (N, H, W) or (N, H, W, C) -> float (N, C, H, W) in [-1, 1]."""
    x = torch.as_tensor(np.asarray(images))
    x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
    return x.float().div(127.5).sub(1.0).contiguous()


def load_dataset(name: str, data_dir: str = "data", train: bool = True, download: bool = True,
                 n: int = 0, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(images in [-1, 1] as (N, C, H, W), labels (N,), number of classes). `n` > 0 keeps a random subset."""
    root = Path(data_dir)
    if name == "shapes":  # generated, not downloaded; a subset is generated directly
        imgs, labels = make_shapes(n or (12000 if train else 2000), seed=(0 if train else 1) + 10 * seed)
        n = 0
    else:
        from torchvision import datasets as tvd

        klass = {"mnist": tvd.MNIST, "fashion-mnist": tvd.FashionMNIST, "cifar10": tvd.CIFAR10}.get(name)
        if klass is None:
            raise ValueError(f"unknown dataset {name!r}: {DATASETS}")
        ds = klass(str(root / name), train=train, download=download)
        imgs, labels = ds.data, np.asarray(ds.targets)
    x, y = _to_tensor(imgs), torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if n and n < len(x):
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:n]
        x, y = x[idx], y[idx]
    return x, y, len(CLASS_NAMES[name])


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving / plotting."""
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Download a dataset into data/ (or generate shapes) and preview it.")
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--preview", default=None, help="PNG path for an 8 x 8 grid (default data/<name>_preview.png)")
    a = p.parse_args(argv)
    x, _y, k = load_dataset(a.dataset, a.data_dir, train=True)
    xt, _, _ = load_dataset(a.dataset, a.data_dir, train=False)
    print(f"{a.dataset}: train {tuple(x.shape)}, test {tuple(xt.shape)}, {k} classes, "
          f"pixel range [{x.min():.0f}, {x.max():.0f}], mean {x.mean():.3f}, std {x.std():.3f}")
    from diffusion.viz import save_grid

    out = a.preview or str(Path(a.data_dir) / f"{a.dataset}_preview.png")
    save_grid(x[:64], out, nrow=8)
    print("preview ->", out)


if __name__ == "__main__":
    main()
