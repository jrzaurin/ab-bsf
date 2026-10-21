from typing import Any, Dict, Sequence

import numpy as np

from ab_bsf.bayesian.models.base import ConjugateModel, Data, Statistics

WEAK = 0.1


class _GammaModel(ConjugateModel):
    """Shared part of the models with a Gamma prior on a rate."""

    def __init__(self, a_prior: float = WEAK, b_prior: float = WEAK) -> None:
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        self.prior: Dict[str, Any] = {"a_prior": a_prior, "b_prior": b_prior}

    def validate(self, statistics: Statistics) -> None:
        if statistics["totals"] <= 0:
            raise ValueError(
                "Input variable 'totals' is expected to be positive integer."
            )
        if statistics["sum_values"] < 0:
            raise ValueError(
                "Input variable 'sum_values' is expected to be non-negative number."
            )

    def n_observations(self, statistics: Statistics) -> float:
        return statistics["totals"]

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        totals, total_value = statistics["totals"], statistics["sum_values"]
        return {
            "totals": totals,
            "sum_values": round(total_value, 5),
            "observed_average": round(total_value / totals, 5),
        }

    def _summarize_non_negative(self, data: Data) -> np.ndarray:
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if (x < 0).any():
            raise ValueError("Input data needs to be a list of non-negative numbers.")
        return x


class GammaPoisson(_GammaModel):
    """
    Gamma-Poisson conjugate model for counts (trades per session, sessions per
    user). The compared metric is the Poisson rate.

    - Likelihood: each observation is Poisson(rate). The data enters only through
      ``totals`` (number of observations) and ``sum_values`` (total count).
    - Prior: ``rate ~ Gamma(a_prior, b_prior)`` with ``b_prior`` as the rate
      parameter. Read it as ``a_prior`` events seen over ``b_prior`` observations
      before the experiment (default 0.1 and 0.1, very weak).
    - Posterior: ``rate | data ~ Gamma(a_prior + sum_values, b_prior + totals)``,
      with mean ``(a_prior + sum_values) / (b_prior + totals)``.

    Note: heavily over-dispersed counts (variance >> mean) make the Poisson
    posterior too narrow; check var/mean before trusting the intervals.
    """

    def summarize(self, data: Data) -> Statistics:
        x = self._summarize_non_negative(data)
        if not np.all(np.mod(x, 1) == 0):
            raise ValueError("Input data needs to be non-negative integers.")
        return {"totals": int(x.size), "sum_values": float(x.sum())}

    def fit_priors(self, historical: Data) -> "GammaPoisson":
        """The posterior of the history under the default Gamma(0.1, 0.1) prior."""
        statistics = self.summarize(historical)
        self.prior = {
            "a_prior": statistics["sum_values"] + WEAK,
            "b_prior": statistics["totals"] + WEAK,
        }
        return self

    def weighted(self, weight: float) -> "GammaPoisson":
        # b_prior is the exposure, in observations
        scale = weight / self.prior["b_prior"]
        return GammaPoisson(
            self.prior["a_prior"] * scale, self.prior["b_prior"] * scale
        )

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        return rng.gamma(
            a + statistics["sum_values"], 1 / (b + statistics["totals"]), sim_count
        )

    def posterior_mean(self, statistics: Statistics) -> float:
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        return (a + statistics["sum_values"]) / (b + statistics["totals"])


class GammaExponential(_GammaModel):
    """
    Gamma-Exponential conjugate model for positive durations (session time, time to
    first trade). The compared metric is the mean duration, ``1 / rate``. Use
    ``evaluate(min_is_best=True)`` when shorter is better.

    - Likelihood: each observation is Exponential(rate). The data enters only
      through ``totals`` (number of observations) and ``sum_values`` (total time).
    - Prior: ``rate ~ Gamma(a_prior, b_prior)`` with ``b_prior`` as the rate
      parameter. Read it as ``a_prior`` events seen over ``b_prior`` units of time
      before the experiment (default 0.1 and 0.1, very weak).
    - Posterior: ``rate | data ~ Gamma(a_prior + totals, b_prior + sum_values)``.
      The mean duration ``1 / rate`` is then Inverse-Gamma, with mean
      ``(b_prior + sum_values) / (a_prior + totals - 1)``.

    Conjugacy means the posterior stays in the prior's family, so the update has a
    closed form. Variants are compared by drawing the rate from each posterior and
    inverting it.
    """

    def summarize(self, data: Data) -> Statistics:
        x = self._summarize_non_negative(data)
        return {"totals": int(x.size), "sum_values": float(x.sum())}

    def fit_priors(self, historical: Data) -> "GammaExponential":
        """The posterior of the history under the default Gamma(0.1, 0.1) prior."""
        statistics = self.summarize(historical)
        self.prior = {
            "a_prior": statistics["totals"] + WEAK,
            "b_prior": statistics["sum_values"] + WEAK,
        }
        return self

    def weighted(self, weight: float) -> "GammaExponential":
        # a_prior is the number of events
        scale = weight / self.prior["a_prior"]
        return GammaExponential(
            self.prior["a_prior"] * scale, self.prior["b_prior"] * scale
        )

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        rate = rng.gamma(
            a + statistics["totals"], 1 / (b + statistics["sum_values"]), sim_count
        )
        return np.reciprocal(rate)

    def posterior_mean(self, statistics: Statistics) -> float:
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        return (b + statistics["sum_values"]) / (a + statistics["totals"] - 1)
