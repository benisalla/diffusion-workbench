"""Evaluation shared by every method: `diffusion-eval`.

1. A small CNN classifier is trained once per dataset (about a minute on MNIST, ~99% test accuracy)
   and cached in checkpoints/classifier_<dataset>.pt.
2. **Classifier-feature FID**: its 64-d penultimate features are computed for the 10 000 test images
   and for N generated images; the Frechet distance between the two Gaussians is the score
   (lower = closer to the data). This is *not* Inception FID: it is only meaningful for comparing
   methods inside this repo on the same dataset. "real vs real" (N training images vs the test set)
   gives the floor that a perfect model would reach with the same N.
3. For class-conditional models: **label accuracy**, the fraction of samples generated for class c that
   the classifier recognises as c (how well the conditioning / guidance works).
4. **NFE sweep**: the same checkpoint sampled with 5, 10, 20, 50, 100 network evaluations per image,
   for each of the method's samplers, gives the quality-vs-cost curve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusion.data import load_dataset
from diffusion.metrics import fid_from_features
from diffusion.sample import NFECounter, generate, load_checkpoint

DEFAULT_NFES = (5, 10, 20, 50, 100)


class Classifier(nn.Module):
    """3 conv layers + global average pooling -> 64-d features -> class logits."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, width: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.BatchNorm2d(4 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def train_classifier(dataset: str, data_dir: str = "data", steps: int = 1500, batch_size: int = 128,
                     seed: int = 0, data=None, verbose: bool = True) -> tuple[Classifier, float]:
    """Train the feature classifier; returns (model, test accuracy). `data` = ((x, y), (xt, yt), k) for tests."""
    torch.manual_seed(seed)
    if data is None:
        x, y, k = load_dataset(dataset, data_dir, train=True)
        xt, yt, _ = load_dataset(dataset, data_dir, train=False)
    else:
        (x, y), (xt, yt), k = data
    clf = Classifier(x.shape[1], k)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)
    clf.train()
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch_size,), generator=gen)
        xb = x[idx]
        shift = torch.randint(-2, 3, (2,), generator=gen).tolist()  # small random translation
        xb = torch.roll(xb, shifts=shift, dims=(2, 3))
        loss = F.cross_entropy(clf(xb), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step + 1) % 500 == 0:
            print(f"classifier step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)
    clf.eval()
    acc = (predict(clf, xt) == yt).float().mean().item()
    return clf, acc


@torch.no_grad()
def predict(clf: Classifier, x: torch.Tensor, batch: int = 1000) -> torch.Tensor:
    return torch.cat([clf(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)])


@torch.no_grad()
def features(clf: Classifier, x: torch.Tensor, batch: int = 1000) -> np.ndarray:
    return torch.cat([clf.features(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()


def get_classifier(dataset: str, data_dir: str = "data", cache_dir: str = "checkpoints") -> Classifier:
    """Load the cached classifier for this dataset, training it first if needed."""
    path = Path(cache_dir) / f"classifier_{dataset}.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        clf = Classifier(ck["in_channels"], ck["num_classes"])
        clf.load_state_dict(ck["state"])
        return clf.eval()
    print(f"training the {dataset} feature classifier (once) -> {path}", flush=True)
    clf, acc = train_classifier(dataset, data_dir)
    print(f"classifier test accuracy {acc:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": clf.state_dict(), "in_channels": clf.body[0].in_channels,
                "num_classes": clf.head.out_features, "test_acc": acc}, path)
    return clf


class Evaluator:
    """Holds the classifier and the test-set feature statistics, scores batches of samples."""

    def __init__(self, dataset: str, data_dir: str = "data", cache_dir: str = "checkpoints", clf=None,
                 test=None):
        self.clf = clf if clf is not None else get_classifier(dataset, data_dir, cache_dir)
        xt, _yt, _ = test if test is not None else load_dataset(dataset, data_dir, train=False)
        self.test_feats = features(self.clf, xt)
        self.dataset, self.data_dir = dataset, data_dir

    def fid(self, x: torch.Tensor) -> float:
        return fid_from_features(features(self.clf, x), self.test_feats)

    def label_accuracy(self, x: torch.Tensor, labels: torch.Tensor) -> float:
        return (predict(self.clf, x) == labels).float().mean().item()

    def real_floor(self, n: int, seed: int = 0) -> float:
        """FID of n real training images against the test set: the best score reachable with n samples."""
        x, _, _ = load_dataset(self.dataset, self.data_dir, train=True, n=n, seed=seed)
        return self.fid(x)


def evaluate_checkpoint(path: str, evaluator: Evaluator, nfes=DEFAULT_NFES, samplers=None, n_samples: int = 1000,
                        guidance: float = 1.0, seed: int = 0, verbose: bool = True) -> list[dict]:
    """NFE sweep of one checkpoint: one row per (sampler, nfe) with FID, label accuracy and timing."""
    net, method, cfg = load_checkpoint(path)
    k = cfg.model.num_classes
    labels = torch.arange(n_samples) % k if k else None
    rows = []
    for sampler in samplers or method.samplers:
        for nfe in nfes:
            steps = method.steps_for_nfe(sampler, nfe)
            counter = NFECounter(net)
            t0 = time.time()
            x = generate(counter, method, n_samples, steps, sampler, labels, guidance, seed, batch_size=n_samples)
            seconds = time.time() - t0
            row = {"sampler": sampler, "steps": steps, "nfe": counter.calls, "guidance": guidance,
                   "fid": evaluator.fid(x), "sec_per_100": 100 * seconds / n_samples}
            if k:
                row["label_acc"] = evaluator.label_accuracy(x, labels)
            rows.append(row)
            if verbose:
                extra = f"  label acc {row['label_acc']:.3f}" if k else ""
                print(f"{sampler:6s} steps {steps:4d} NFE {row['nfe']:4d}  FID {row['fid']:8.3f}{extra}  "
                      f"{row['sec_per_100']:.2f} s / 100 images", flush=True)
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Classifier-feature FID / label accuracy of a checkpoint vs NFE.")
    p.add_argument("checkpoint", nargs="?", help="a last.pt from diffusion-train (omit with --train-classifier)")
    p.add_argument("--nfe", type=int, nargs="+", default=list(DEFAULT_NFES))
    p.add_argument("--samplers", nargs="+", default=None, help="default: every sampler of the method")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0], help="CFG scales to evaluate")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default="checkpoints", help="where the feature classifier is cached")
    p.add_argument("--train-classifier", default=None, metavar="DATASET", help="only train/cache the classifier")
    p.add_argument("--out", default=None, help="results JSON (default: next to the checkpoint)")
    p.add_argument("--plot", default=None, help="NFE curve PNG (default: next to the checkpoint)")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.train_classifier:
        get_classifier(a.train_classifier, a.data_dir, a.cache_dir)
        return
    if not a.checkpoint:
        p.error("give a checkpoint, or --train-classifier DATASET")
    _, _, cfg = load_checkpoint(a.checkpoint)
    ev = Evaluator(cfg.train.dataset, a.data_dir, a.cache_dir)
    floor = ev.real_floor(a.n_samples)
    print(f"real vs real floor (N={a.n_samples}): FID {floor:.3f}")
    rows = []
    for w in a.guidance:
        rows += evaluate_checkpoint(a.checkpoint, ev, a.nfe, a.samplers, a.n_samples, w)
    folder = Path(a.checkpoint).parent
    out = Path(a.out or folder / "eval.json")
    out.write_text(json.dumps({"checkpoint": a.checkpoint, "floor": floor, "n_samples": a.n_samples, "rows": rows},
                              indent=1))
    from diffusion.viz import plot_nfe_curves

    curves: dict = {}
    for r in rows:
        curves.setdefault(f"{r['sampler']} w={r['guidance']:g}", []).append((r["nfe"], r["fid"]))
    plot_nfe_curves(curves, a.plot or folder / "nfe.png", floor=floor)
    print("results ->", out)


if __name__ == "__main__":
    main()
