"""Frechet distance between two Gaussians fitted to feature vectors (the formula behind FID).

    FD = ||mu_1 - mu_2||^2 + Tr(S_1 + S_2 - 2 (S_1 S_2)^(1/2))

Tr((S_1 S_2)^(1/2)) is computed as the sum of square roots of the eigenvalues of
S_1^(1/2) S_2 S_1^(1/2), which is symmetric positive semi-definite, so a plain `eigh` is enough
(no scipy needed).

Here the features come from a small classifier trained on the same dataset (see evaluate.py), not
from Inception-v3: the numbers are comparable *between methods in this repo*, not with published FIDs.
"""

from __future__ import annotations

import numpy as np


def gaussian_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(0), np.cov(feats, rowvar=False)


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet_distance(mu1, s1, mu2, s2) -> float:
    r1 = _sqrtm_psd(s1)
    mid = r1 @ s2 @ r1
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((mid + mid.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    return frechet_distance(*gaussian_stats(f1), *gaussian_stats(f2))
