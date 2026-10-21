import copy
from typing import Any, Dict, Tuple

import numpy as np

from ab_bsf.bayesian.models.base import ConjugateModel, Data, Statistics
from ab_bsf.bayesian.models.beta_bernoulli import JEFFREYS, BetaBernoulli
from ab_bsf.bayesian.models.normal_inverse_gamma import NormalInverseGamma


class DeltaNormal(ConjugateModel):
    """
    Model for zero-inflated data whose non-zero values can be positive or negative
    (P&L, net deposits), made of two conjugate models.

    Each variant's data is split in two parts, modelled independently:

    1. Whether an observation is non-zero, with a ``BetaBernoulli`` model:
       ``non_zeros`` out of ``totals`` is Binomial(totals, p), the prior is
       ``p ~ Beta(a_prior_beta, b_prior_beta)`` and the posterior is
       ``Beta(a_prior_beta + non_zeros, b_prior_beta + totals - non_zeros)``.
    2. The size of the non-zero values, with a ``NormalInverseGamma`` model:
       they are Normal(mu, sigma^2) with both parameters unknown, the prior is
       ``sigma^2 ~ InverseGamma(a_prior_ig, b_prior_ig)`` and
       ``mu | sigma^2 ~ Normal(m_prior, sigma^2 / w_prior)``, where ``w_prior`` is
       the prior's weight in pseudo-observations. The posterior has the same form.

    The compared metric is the mean per observation, ``p x mu``. Variants are
    compared by drawing from both posteriors and multiplying the draws. The
    ``posterior_mean`` is exact: E[p] x E[mu], since both parts are independent.

    When a prior is weighted, the Beta part is worth ``weight`` observations and
    the Normal part ``weight`` times the prior's share of non-zero values.
    """

    count_key = "non_zeros"
    magnitude_keys = ("sum_values", "sum_values_2")

    def __init__(
        self,
        a_prior_beta: float = JEFFREYS,
        b_prior_beta: float = JEFFREYS,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
    ) -> None:
        self.presence = BetaBernoulli(a_prior_beta, b_prior_beta)
        self.magnitude = NormalInverseGamma(m_prior, a_prior_ig, b_prior_ig, w_prior)

    @property
    def prior(self) -> Dict[str, Any]:
        return {
            "a_prior_beta": self.presence.prior["a_prior"],
            "b_prior_beta": self.presence.prior["b_prior"],
            **self.magnitude.prior,
        }

    def summarize(self, data: Data) -> Statistics:
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        non_zero = x[x != 0]
        return {
            "totals": int(x.size),
            "non_zeros": int(non_zero.size),
            "sum_values": float(non_zero.sum()),
            "sum_values_2": float((non_zero**2).sum()),
        }

    def validate(self, statistics: Statistics) -> None:
        self.presence.validate(self._parts(statistics)[0])

    def fit_priors(self, historical: Data) -> "DeltaNormal":
        """Fit the Beta part to the non-zero share and the Normal part to the values."""
        x = np.asarray(historical, dtype=float)
        non_zero = self._non_zero(x)
        self.magnitude.fit_priors(self._magnitude_values(x[non_zero]))
        self.presence.fit_priors(non_zero.astype(int))
        return self

    def weighted(self, weight: float) -> "DeltaNormal":
        a, b = self.presence.prior["a_prior"], self.presence.prior["b_prior"]
        weighted = copy.copy(self)
        weighted.presence = self.presence.weighted(weight)
        weighted.magnitude = self.magnitude.weighted(weight * a / (a + b))
        return weighted

    def n_observations(self, statistics: Statistics) -> float:
        return statistics["totals"]

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        totals, non_zeros = statistics["totals"], statistics["non_zeros"]
        total_value = statistics["sum_values"]
        return {
            "totals": totals,
            "non_zeros": non_zeros,
            "sum_values": round(total_value, 5),
            "avg_values": round(total_value / totals, 5),
            "avg_non_zero_values": (
                round(total_value / non_zeros, 5) if non_zeros else 0.0
            ),
        }

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        presence, magnitude = self._parts(statistics)
        non_zero_rate = self.presence.posterior_samples(presence, sim_count, rng)
        if magnitude["totals"] == 0:
            return np.zeros(sim_count)
        return non_zero_rate * self._magnitude_samples(magnitude, sim_count, rng)

    def posterior_mean(self, statistics: Statistics) -> float:
        presence, magnitude = self._parts(statistics)
        if magnitude["totals"] == 0:
            return 0.0
        return self.presence.posterior_mean(presence) * self._magnitude_mean(magnitude)

    def _non_zero(self, x: np.ndarray) -> np.ndarray:
        return x != 0

    def _magnitude_values(self, non_zero_values: np.ndarray) -> np.ndarray:
        return non_zero_values

    def _magnitude_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        return self.magnitude.posterior_samples(statistics, sim_count, rng)

    def _magnitude_mean(self, statistics: Statistics) -> float:
        return self.magnitude.posterior_mean(statistics)

    def _parts(self, statistics: Statistics) -> Tuple[Statistics, Statistics]:
        """Statistics of the Beta part and of the Normal part."""
        non_zeros = statistics[self.count_key]
        sums, sums_2 = (statistics[key] for key in self.magnitude_keys)
        return (
            {"totals": statistics["totals"], "positives": non_zeros},
            {"totals": non_zeros, "sum_values": sums, "sum_values_2": sums_2},
        )


class DeltaLognormal(DeltaNormal):
    """
    Model for zero-inflated, right-skewed data (revenue or volume per session), made
    of two conjugate models.

    Like ``DeltaNormal``, each variant's data is split in a Beta part (whether an
    observation is non-zero, with ``positives`` out of ``totals``) and a
    Normal-Inverse-Gamma part, but the non-zero values are assumed lognormal, so
    the Normal-Inverse-Gamma model applies to their logs: ``mu`` and ``sigma^2``
    are the mean and variance of ``log(value)``, and ``m_prior`` is in log space
    (``m_prior=1`` means a geometric mean of e).

    The compared metric is the mean per observation, which combines both parts:
    ``p x exp(mu + sigma^2 / 2)``, i.e. P(non-zero) x E[value | non-zero]. Variants
    are compared by drawing from both posteriors and multiplying the draws.

    The ``posterior_mean`` is a closed form, ``P(non-zero) x exp(m_post + b_post /
    (2 * a_post))``, where the variance is a plug-in point estimate. Averaging the
    draws instead is unstable with few non-zero values, since E[exp(sigma^2 / 2)]
    is infinite when sigma^2 is Inverse-Gamma.

    Parameters
    ----------
    min_positive : values <= min_positive count as zeros, e.g. to drop dust trades
        that would otherwise create a second mode in log space.
    """

    count_key = "positives"
    magnitude_keys = ("sum_logs", "sum_logs_2")

    def __init__(
        self,
        a_prior_beta: float = JEFFREYS,
        b_prior_beta: float = JEFFREYS,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        min_positive: float = 0.0,
    ) -> None:
        super().__init__(
            a_prior_beta, b_prior_beta, m_prior, a_prior_ig, b_prior_ig, w_prior
        )
        self.min_positive = min_positive

    def summarize(self, data: Data) -> Statistics:
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if (x < 0).any():
            raise ValueError("Input data needs to be a list of non-negative numbers.")
        positives = x[self._non_zero(x)]
        logs = np.log(positives)
        return {
            "totals": int(x.size),
            "positives": int(positives.size),
            "sum_values": float(positives.sum()),
            "sum_logs": float(logs.sum()),
            "sum_logs_2": float((logs**2).sum()),
        }

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        totals, positives = statistics["totals"], statistics["positives"]
        total_value = statistics["sum_values"]
        return {
            "totals": totals,
            "positives": positives,
            "sum_values": round(total_value, 5),
            "avg_values": round(total_value / totals, 5),
            "avg_positive_values": (
                round(total_value / positives, 5) if positives else 0.0
            ),
        }

    def _non_zero(self, x: np.ndarray) -> np.ndarray:
        return x > self.min_positive

    def _magnitude_values(self, non_zero_values: np.ndarray) -> np.ndarray:
        return np.log(non_zero_values)

    def _magnitude_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        mu, sigma_2 = self.magnitude.posterior_draws(statistics, sim_count, rng)
        return np.exp(mu + sigma_2 / 2)

    def _magnitude_mean(self, statistics: Statistics) -> float:
        m_post, _, a_post, b_post = self.magnitude.posterior_parameters(statistics)
        return float(np.exp(m_post + b_post / (2 * a_post)))
