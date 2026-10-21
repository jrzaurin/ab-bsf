from typing import List

import numpy as np

__all__ = [
    "estimate_probabilities",
    "estimate_expected_loss",
    "estimate_credible_intervals",
    "estimate_relative_differences",
    "estimate_relative_credible_intervals",
]

_EPS = 1e-12


# Summaries of posterior draws
def estimate_probabilities(data: np.ndarray, min_is_best: bool = False) -> List[float]:
    """P(variant is best) for each row of ``data``."""
    data = np.asarray(data)
    best = np.argmin(data, axis=0) if min_is_best else np.argmax(data, axis=0)
    counts = np.bincount(best, minlength=data.shape[0])
    return [float(x) for x in np.round(counts / data.shape[1], 7)]


def estimate_expected_loss(data: np.ndarray, min_is_best: bool = False) -> List[float]:
    """E[best - variant] (or E[variant - best] when min_is_best) for each row."""
    data = np.asarray(data)
    best = np.min(data, axis=0) if min_is_best else np.max(data, axis=0)
    return [float(x) for x in np.abs(np.mean(best - data, axis=1)).round(7)]


def estimate_credible_intervals(
    data: np.ndarray, alpha: float = 0.95
) -> List[List[float]]:
    """Equal-tailed credible interval of probability ``alpha`` for each row."""
    if not 0 <= alpha <= 1:
        raise ValueError(
            "Credible interval's probability alpha has to be between 0 and 1."
        )
    lo, hi = (1 - alpha) / 2, (1 + alpha) / 2
    return np.round(np.quantile(np.asarray(data), [lo, hi], axis=1).T, 7).tolist()


def _relative_draws(data: np.ndarray, reference_idx: int) -> np.ndarray:
    ref = data[reference_idx]
    ref_safe = np.where(np.abs(ref) < _EPS, _EPS, ref)
    return (data - ref) / np.abs(ref_safe)


def estimate_relative_differences(
    data: np.ndarray, reference_idx: int = 0
) -> List[float]:
    """
    Posterior mean of (variant - reference) / |reference| for each row
    (0 for the reference itself).
    """
    rel = _relative_draws(np.asarray(data, dtype=float), reference_idx)
    out = [float(x) for x in np.round(rel.mean(axis=1), 7)]
    out[reference_idx] = 0.0
    return out


def estimate_relative_credible_intervals(
    data: np.ndarray, reference_idx: int = 0, alpha: float = 0.95
) -> List[List[float]]:
    """Credible intervals for the relative difference vs the reference."""
    rel = _relative_draws(np.asarray(data, dtype=float), reference_idx)
    out = estimate_credible_intervals(rel, alpha)
    out[reference_idx] = [0.0, 0.0]
    return out
