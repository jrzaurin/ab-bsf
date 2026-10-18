"""
Posterior samplers for the conjugate models.

Adapted from Matus Baniar's bayesian-testing (MIT licence,
https://github.com/Matt52/bayesian-testing), which ab_bsf mirrored.
"""

from typing import List, Tuple, Union

import numpy as np

Seed = Union[int, np.random.SeedSequence, None]

__all__ = [
    "beta_posteriors_all",
    "normal_posteriors",
    "lognormal_posteriors",
    "dirichlet_posteriors",
    "pois_gamma_posteriors_all",
    "exp_gamma_posteriors_all",
]


def beta_posteriors_all(
    totals: List[int],
    positives: List[int],
    sim_count: int,
    a_priors_beta: List[float],
    b_priors_beta: List[float],
    seed: Seed = None,
) -> np.ndarray:
    """
    Beta(a + positives, b + totals - positives) draws for every variant.

    Returns an array of shape (n_variants, sim_count).
    """
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.beta(
                positives[i] + a_priors_beta[i],
                totals[i] - positives[i] + b_priors_beta[i],
                sim_count,
            )
            for i in range(len(totals))
        ]
    )


def normal_posteriors(
    total: int,
    sums: float,
    sums_2: float,
    sim_count: int = 20000,
    prior_m: float = 1,
    prior_a: float = 0,
    prior_b: float = 0,
    prior_w: float = 0.01,
    seed: Seed = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw (mu, sigma^2) from the Normal-Inverse-Gamma posterior for aggregated normal data.

    Parameters
    ----------
    total : number of observations.
    sums, sums_2 : sum and sum of squares of the data.
    prior_m : prior mean of mu.
    prior_a, prior_b : Inverse-Gamma prior on sigma^2 (0 is allowed since n >= 1).
    prior_w : prior effective sample size for mu.

    Returns
    -------
    mu_post, sig_2_post : arrays of length sim_count.
    """
    rng = np.random.default_rng(seed)
    x_bar = sums / total
    a_post = prior_a + total / 2
    b_post = (
        prior_b
        + 0.5 * (sums_2 - 2 * sums * x_bar + total * x_bar**2)
        + (total * prior_w) / (2 * (total + prior_w)) * (x_bar - prior_m) ** 2
    )
    # numpy's gamma takes a scale, hence 1 / b_post
    sig_2_post = 1 / rng.gamma(a_post, 1 / b_post, sim_count)
    m_post = (total * x_bar + prior_w * prior_m) / (total + prior_w)
    mu_post = rng.normal(m_post, np.sqrt(sig_2_post / (total + prior_w)))
    return mu_post, sig_2_post


def lognormal_posteriors(
    total: int,
    sum_logs: float,
    sum_logs_2: float,
    sim_count: int = 20000,
    prior_m: float = 1,
    prior_a: float = 0,
    prior_b: float = 0,
    prior_w: float = 0.01,
    seed: Seed = None,
) -> np.ndarray:
    """
    Posterior draws of the lognormal *mean* exp(mu + sigma^2 / 2), using the
    normal posterior of the log data.
    """
    if total <= 0:
        return np.zeros(sim_count)
    mu, sig_2 = normal_posteriors(
        total, sum_logs, sum_logs_2, sim_count, prior_m, prior_a, prior_b, prior_w, seed
    )
    return np.exp(mu + sig_2 / 2)


def dirichlet_posteriors(
    concentration: List[int],
    prior: List[float],
    sim_count: int = 20000,
    seed: Seed = None,
) -> np.ndarray:
    """Dirichlet posterior draws for one variant, shape (sim_count, n_states)."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet([p + c for p, c in zip(prior, concentration)], sim_count)


def pois_gamma_posteriors_all(
    totals: List[int],
    sums: List[float],
    sim_count: int,
    a_priors_gamma: List[float],
    b_priors_gamma: List[float],
    seed: Seed = None,
) -> np.ndarray:
    """Gamma(a + sum, rate=b + n) draws of the Poisson rate for every variant."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.gamma(
                sums[i] + a_priors_gamma[i],
                1 / (totals[i] + b_priors_gamma[i]),
                sim_count,
            )
            for i in range(len(totals))
        ]
    )


def exp_gamma_posteriors_all(
    totals: List[int],
    sums: List[float],
    sim_count: int,
    a_priors_gamma: List[float],
    b_priors_gamma: List[float],
    seed: Seed = None,
) -> np.ndarray:
    """Gamma(a + n, rate=b + sum) draws of the Exponential *rate* for every variant."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.gamma(
                totals[i] + a_priors_gamma[i],
                1 / (sums[i] + b_priors_gamma[i]),
                sim_count,
            )
            for i in range(len(totals))
        ]
    )
