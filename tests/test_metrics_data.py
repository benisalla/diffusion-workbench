import numpy as np
import torch

from diffusion.data import load_dataset, make_shapes, to_unit
from diffusion.metrics import fid_from_features, frechet_distance


def test_frechet_distance_of_known_gaussians():
    mu, s = np.zeros(3), np.eye(3)
    assert abs(frechet_distance(mu, s, mu, s)) < 1e-8
    assert abs(frechet_distance(mu, s, mu + 2, s) - 12) < 1e-8          # ||mu1 - mu2||^2 = 3 * 4
    # 1-d: (s1 - s2)^2 for standard deviations s1, s2
    assert abs(frechet_distance(np.zeros(1), np.eye(1) * 4, np.zeros(1), np.eye(1)) - 1) < 1e-8


def test_fid_same_distribution_is_small_and_shift_is_large():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4000, 8)), rng.normal(size=(4000, 8))
    assert fid_from_features(a, b) < 0.05
    assert fid_from_features(a, b + 1) > 7.5


def test_shapes_dataset():
    x, y = make_shapes(40, seed=3)
    x2, _ = make_shapes(40, seed=3)
    assert x.shape == (40, 28, 28) and x.dtype == np.uint8 and (x == x2).all()
    assert set(y.tolist()) <= {0, 1, 2, 3} and (x.reshape(40, -1).max(1) > 0).all()
    xs, ys, k = load_dataset("shapes", train=False, n=10)
    assert xs.shape == (10, 1, 28, 28) and k == 4 and xs.min() >= -1 and xs.max() <= 1
    assert to_unit(xs).min() >= 0 and to_unit(xs).max() <= 1 and ys.dtype == torch.long
