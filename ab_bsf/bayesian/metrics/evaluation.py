"""
Turning posterior draws into decision metrics.

Two layers:

* ``sample_*_agg`` functions return the posterior draws of each variant's mean
  metric, shape (n_variants, sim_count).
* ``estimate_*`` functions summarise any such array: probability of being best,
  expected loss, credible intervals, and (ab_bsf additions) relative differences
  vs a reference variant with their credible intervals.
"""

from typing import List, Optional

import numpy as np

from ab_bsf.bayesian.metrics.posteriors import (
    beta_posteriors_all,
    dirichlet_posteriors,
    exp_gamma_posteriors_all,
    lognormal_posteriors,
    normal_posteriors,
    pois_gamma_posteriors_all,
)

__all__ = [
    "estimate_probabilities",
    "estimate_expected_loss",
    "estimate_credible_intervals",
    "estimate_relative_differences",
    "estimate_relative_credible_intervals",
    "sample_bernoulli_agg",
    "sample_normal_agg",
    "sample_delta_lognormal_agg",
    "sample_delta_normal_agg",
    "sample_poisson_agg",
    "sample_exponential_agg",
    "sample_numerical_dirichlet_agg",
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


def _fill(values: Optional[List], default, n: int) -> List:
    return list(values) if values else [default] * n


# Posterior draws of the mean metric per variant
def sample_bernoulli_agg(
    totals,
    positives,
    a_priors_beta=None,
    b_priors_beta=None,
    sim_count=20000,
    seed=None,
) -> np.ndarray:
    """Conversion-rate draws, Beta-Bernoulli. Default prior Beta(0.5, 0.5)."""
    if len(totals) != len(positives):
        raise ValueError("totals and positives need to have the same length")
    n = len(totals)
    return beta_posteriors_all(
        totals,
        positives,
        sim_count,
        _fill(a_priors_beta, 0.5, n),
        _fill(b_priors_beta, 0.5, n),
        seed,
    )


def sample_normal_agg(
    totals,
    sums,
    sums_2,
    m_priors=None,
    a_priors_ig=None,
    b_priors_ig=None,
    w_priors=None,
    sim_count=20000,
    seed=None,
) -> np.ndarray:
    """Mean draws, Normal with unknown variance (Normal-Inverse-Gamma)."""
    n = len(totals)
    m, a, b, w = (
        _fill(m_priors, 1, n),
        _fill(a_priors_ig, 0, n),
        _fill(b_priors_ig, 0, n),
        _fill(w_priors, 0.01, n),
    )
    seeds = np.random.SeedSequence(seed).spawn(n)
    return np.array(
        [
            normal_posteriors(
                totals[i],
                sums[i],
                sums_2[i],
                sim_count,
                m[i],
                a[i],
                b[i],
                w[i],
                seeds[i],
            )[0]
            for i in range(n)
        ]
    )


def sample_delta_lognormal_agg(
    totals,
    non_zeros,
    sum_logs,
    sum_logs_2,
    a_priors_beta=None,
    b_priors_beta=None,
    m_priors=None,
    a_priors_ig=None,
    b_priors_ig=None,
    w_priors=None,
    sim_count=20000,
    seed=None,
) -> np.ndarray:
    """Mean-per-observation draws: P(non-zero) x lognormal mean of the non-zeros."""
    n = len(totals)
    seeds = np.random.SeedSequence(seed).spawn(n + 1)
    beta = beta_posteriors_all(
        totals,
        non_zeros,
        sim_count,
        _fill(a_priors_beta, 0.5, n),
        _fill(b_priors_beta, 0.5, n),
        seeds[0],
    )
    m, a, b, w = (
        _fill(m_priors, 1, n),
        _fill(a_priors_ig, 0, n),
        _fill(b_priors_ig, 0, n),
        _fill(w_priors, 0.01, n),
    )
    logn = np.array(
        [
            lognormal_posteriors(
                non_zeros[i],
                sum_logs[i],
                sum_logs_2[i],
                sim_count,
                m[i],
                a[i],
                b[i],
                w[i],
                seeds[i + 1],
            )
            for i in range(n)
        ]
    )
    return beta * logn


def sample_delta_normal_agg(
    totals,
    non_zeros,
    sums,
    sums_2,
    a_priors_beta=None,
    b_priors_beta=None,
    m_priors=None,
    a_priors_ig=None,
    b_priors_ig=None,
    w_priors=None,
    sim_count=20000,
    seed=None,
) -> np.ndarray:
    """Mean-per-observation draws: P(non-zero) x normal mean of the non-zeros."""
    n = len(totals)
    seeds = np.random.SeedSequence(seed).spawn(n + 1)
    beta = beta_posteriors_all(
        totals,
        non_zeros,
        sim_count,
        _fill(a_priors_beta, 0.5, n),
        _fill(b_priors_beta, 0.5, n),
        seeds[0],
    )
    m, a, b, w = (
        _fill(m_priors, 1, n),
        _fill(a_priors_ig, 0, n),
        _fill(b_priors_ig, 0, n),
        _fill(w_priors, 0.01, n),
    )
    norm = np.array(
        [
            (
                normal_posteriors(
                    non_zeros[i],
                    sums[i],
                    sums_2[i],
                    sim_count,
                    m[i],
                    a[i],
                    b[i],
                    w[i],
                    seeds[i + 1],
                )[0]
                if non_zeros[i] > 0
                else np.zeros(sim_count)
            )
            for i in range(n)
        ]
    )
    return beta * norm


def sample_poisson_agg(
    totals, sums, a_priors_gamma=None, b_priors_gamma=None, sim_count=20000, seed=None
) -> np.ndarray:
    """Poisson-rate draws, Gamma-Poisson. Default prior Gamma(0.1, 0.1)."""
    n = len(totals)
    return pois_gamma_posteriors_all(
        totals,
        sums,
        sim_count,
        _fill(a_priors_gamma, 0.1, n),
        _fill(b_priors_gamma, 0.1, n),
        seed,
    )


def sample_exponential_agg(
    totals, sums, a_priors_gamma=None, b_priors_gamma=None, sim_count=20000, seed=None
) -> np.ndarray:
    """Mean (1 / rate) draws, Gamma-Exponential. Default prior Gamma(0.1, 0.1)."""
    n = len(totals)
    rate = exp_gamma_posteriors_all(
        totals,
        sums,
        sim_count,
        _fill(a_priors_gamma, 0.1, n),
        _fill(b_priors_gamma, 0.1, n),
        seed,
    )
    return np.reciprocal(rate)


def sample_numerical_dirichlet_agg(
    states, concentrations, prior_alphas=None, sim_count=20000, seed=None
) -> np.ndarray:
    """Expected-value draws for categorical numeric outcomes (e.g. 1-5 star ratings)."""
    if not prior_alphas:
        prior_alphas = [[1] * len(states) for _ in concentrations]
    seeds = np.random.SeedSequence(seed).spawn(len(concentrations))
    states = np.asarray(states, dtype=float)
    return np.array(
        [
            dirichlet_posteriors(
                concentrations[i], prior_alphas[i], sim_count, seeds[i]
            )
            @ states
            for i in range(len(concentrations))
        ]
    )
