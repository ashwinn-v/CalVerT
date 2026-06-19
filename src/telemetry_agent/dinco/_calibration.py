"""Calibration metrics: ECE (equal-mass), Brier, AUROC, saturation, separation.

NumPy-only, no torch dependency. Self-tested via `test_calibration.py`.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def _to_arrays(scores, probs):
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if s.shape != p.shape:
        raise ValueError(f"shape mismatch: scores {s.shape} vs probs {p.shape}")
    mask = ~np.isnan(p)
    return s[mask], np.clip(p[mask], 0.0, 1.0)


def ece_equal_mass(scores: Sequence[float], probs: Sequence[float], n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-mass (quantile) bins.

    Equal-mass bins are more stable than equal-width when the confidence
    distribution is skewed (which DINCO outputs typically are).
    """
    s, p = _to_arrays(scores, probs)
    n = len(p)
    if n == 0:
        return float('nan')
    order = np.argsort(p)
    p_sorted = p[order]
    s_sorted = s[order]
    bin_size = n / n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = int(round(b * bin_size))
        hi = int(round((b + 1) * bin_size))
        if hi <= lo:
            continue
        bp = p_sorted[lo:hi]
        bs = s_sorted[lo:hi]
        weight = (hi - lo) / n
        ece += weight * abs(bs.mean() - bp.mean())
    return float(ece)


def brier(scores: Sequence[float], probs: Sequence[float]) -> float:
    s, p = _to_arrays(scores, probs)
    if len(p) == 0:
        return float('nan')
    return float(np.mean((s - p) ** 2))


def auroc(scores: Sequence[float], probs: Sequence[float]) -> float:
    """AUROC of `probs` predicting binary `scores`. NaN if degenerate (all 0 or all 1)."""
    s, p = _to_arrays(scores, probs)
    if len(p) == 0:
        return float('nan')
    pos = s == 1
    neg = s == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # rank-based Mann-Whitney U formulation, ties handled
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def saturation_index(probs: Sequence[float], threshold: float = 0.95) -> float:
    """Fraction of (non-NaN) probabilities at or above the threshold."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return float('nan')
    return float((p >= threshold).mean())


def confidence_separation(scores: Sequence[float], probs: Sequence[float]) -> Dict[str, float]:
    """Mean / median / std of confidence on correct (EM=1) vs incorrect (EM=0)."""
    s, p = _to_arrays(scores, probs)
    if len(p) == 0:
        return {"mean_correct": float('nan'), "mean_incorrect": float('nan'),
                "gap": float('nan'), "n_correct": 0, "n_incorrect": 0}
    p_correct = p[s == 1]
    p_incorrect = p[s == 0]
    return {
        "mean_correct": float(p_correct.mean()) if len(p_correct) else float('nan'),
        "mean_incorrect": float(p_incorrect.mean()) if len(p_incorrect) else float('nan'),
        "gap": float(p_correct.mean() - p_incorrect.mean())
            if (len(p_correct) and len(p_incorrect)) else float('nan'),
        "n_correct": int(len(p_correct)),
        "n_incorrect": int(len(p_incorrect)),
    }


def all_metrics(scores: Sequence[float], probs: Sequence[float], n_bins: int = 15,
                saturation_threshold: float = 0.95) -> Dict[str, float]:
    sep = confidence_separation(scores, probs)
    return {
        "ece": ece_equal_mass(scores, probs, n_bins=n_bins),
        "brier": brier(scores, probs),
        "auroc": auroc(scores, probs),
        "saturation": saturation_index(probs, threshold=saturation_threshold),
        "n": int(np.sum(~np.isnan(np.asarray(probs, dtype=np.float64)))),
        **sep,
    }
