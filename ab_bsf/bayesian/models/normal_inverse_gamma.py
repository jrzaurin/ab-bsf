from typing import Any, Dict, Tuple

import numpy as np

from ab_bsf.bayesian.models.base import ConjugateModel, Data, Statistics


class NormalInverseGamma(ConjugateModel):
    """
    Normal model with unknown mean and variance, Normal-Inverse-Gamma conjugate
    prior. The compared metric is the mean ``mu``.

    - Likelihood: each observation is Normal(mu, sigma^2). The data enters only
      through ``totals``, ``sum_values`` and ``sum_values_2`` (sum of squares), which
      is why it can be passed raw or already aggregated.
    - Prior: ``sigma^2 ~ InverseGamma(a_prior_ig, b_prior_ig)`` and
      ``mu | sigma^2 ~ Normal(m_prior, sigma^2 / w_prior)``. ``m_prior`` is the prior
      mean and ``w_prior`` its weight in pseudo-observations (0.01 is very weak).
      The Inverse-Gamma parameters can be 0 because there is always at least one
      observation.
    - Posterior: same Normal-Inverse-Gamma form, with the data added to the prior.
      The posterior mean of ``mu`` is
      ``(sum_values + w_prior * m_prior) / (totals + w_prior)``, a weighted average
      of the prior mean and the sample mean.

    Conjugacy means the posterior stays in the prior's family, so the update has a
    closed form. This model is also the building block of the delta models.
    """

    def __init__(
        self,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
    ) -> None:
        if a_prior_ig < 0 or b_prior_ig < 0 or w_prior < 0:
            raise ValueError(
                "All priors of [a_ig, b_ig, w] have to be non-negative numbers."
            )
        self.prior: Dict[str, Any] = {
            "m_prior": m_prior,
            "a_prior_ig": a_prior_ig,
            "b_prior_ig": b_prior_ig,
            "w_prior": w_prior,
        }

    def summarize(self, data: Data) -> Statistics:
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        return {
            "totals": int(x.size),
            "sum_values": float(x.sum()),
            "sum_values_2": float((x**2).sum()),
        }

    def validate(self, statistics: Statistics) -> None:
        if statistics["totals"] <= 0:
            raise ValueError(
                "Input variable 'totals' is expected to be positive integer."
            )

    def fit_priors(self, historical: Data) -> "NormalInverseGamma":
        """
        Treat the history as ``n`` observations with its own mean and variance:
        ``w = n``, ``a = n / 2`` and ``b`` half its sum of squared deviations.
        """
        x = np.asarray(historical, dtype=float)
        if x.size < 2:
            raise ValueError("Need at least two observations to fit a variance prior.")
        self.prior = {
            "m_prior": float(x.mean()),
            "a_prior_ig": x.size / 2,
            "b_prior_ig": float(((x - x.mean()) ** 2).sum() / 2),
            "w_prior": float(x.size),
        }
        return self

    def weighted(self, weight: float) -> "NormalInverseGamma":
        """The prior mean stays; its weight and the Inverse-Gamma parameters scale."""
        scale = weight / self.prior["w_prior"]
        return NormalInverseGamma(
            self.prior["m_prior"],
            self.prior["a_prior_ig"] * scale,
            self.prior["b_prior_ig"] * scale,
            weight,
        )

    def n_observations(self, statistics: Statistics) -> float:
        return statistics["totals"]

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        totals, total_value = statistics["totals"], statistics["sum_values"]
        return {
            "totals": totals,
            "sum_values": round(total_value, 5),
            "avg_values": round(total_value / totals, 5),
        }

    def posterior_parameters(
        self, statistics: Statistics
    ) -> Tuple[float, float, float, float]:
        """Posterior ``(m, w, a, b)``, with ``w`` the posterior weight of the mean."""
        n, s, s2 = (
            statistics["totals"],
            statistics["sum_values"],
            statistics["sum_values_2"],
        )
        m, a, b, w = (
            self.prior["m_prior"],
            self.prior["a_prior_ig"],
            self.prior["b_prior_ig"],
            self.prior["w_prior"],
        )
        x_bar = s / n
        b_post = (
            b
            + 0.5 * (s2 - 2 * s * x_bar + n * x_bar**2)
            + (n * w) / (2 * (n + w)) * (x_bar - m) ** 2
        )
        return (s + w * m) / (n + w), n + w, a + n / 2, b_post

    def posterior_draws(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draws of ``(mu, sigma^2)`` from the posterior."""
        m_post, w_post, a_post, b_post = self.posterior_parameters(statistics)
        # numpy's gamma takes a scale, hence 1 / b_post
        sigma_2 = 1 / rng.gamma(a_post, 1 / b_post, sim_count)
        mu = rng.normal(m_post, np.sqrt(sigma_2 / w_post))
        return mu, sigma_2

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.posterior_draws(statistics, sim_count, rng)[0]

    def posterior_mean(self, statistics: Statistics) -> float:
        return self.posterior_parameters(statistics)[0]
