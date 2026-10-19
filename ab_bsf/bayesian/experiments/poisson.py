from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_poisson_agg


class PoissonBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for counts (trades per session, sessions per user),
    Gamma-Poisson conjugate model. The compared metric is the Poisson rate.

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

    @property
    def totals(self):
        return self._values("totals")

    @property
    def sum_values(self):
        return self._values("sum_values")

    @property
    def a_priors(self):
        return self._values("a_prior")

    @property
    def b_priors(self):
        return self._values("b_prior")

    def _posterior_samples(self, sim_count, seed):
        return sample_poisson_agg(
            self.totals, self.sum_values, self.a_priors, self.b_priors, sim_count, seed
        )

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "sum_values": [round(s, 5) for s in self.sum_values],
            "observed_average": [
                round(s / t, 5) for s, t in zip(self.sum_values, self.totals)
            ],
        }

    def _posterior_means(self):
        return [
            round((a + s) / (b + t), 5)
            for s, t, a, b in zip(
                self.sum_values, self.totals, self.a_priors, self.b_priors
            )
        ]

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        sum_values: float,
        a_prior: float = 0.1,
        b_prior: float = 0.1,
        replace: bool = True,
    ) -> None:
        """Add aggregated counts. Default prior Gamma(0.1, 0.1)."""
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0 or sum_values < 0:
            raise ValueError("Need totals > 0 and sum_values >= 0.")
        self._store(
            name,
            {"totals": totals, "sum_values": sum_values},
            {"a_prior": a_prior, "b_prior": b_prior},
            replace,
        )

    def add_variant_data(
        self,
        name: str,
        data: List[int],
        a_prior: float = 0.1,
        b_prior: float = 0.1,
        replace: bool = True,
    ) -> None:
        """Add raw non-negative integer counts."""
        x = np.asarray(data)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if (x < 0).any() or not np.all(np.mod(x, 1) == 0):
            raise ValueError("Input data needs to be non-negative integers.")
        self.add_variant_data_agg(
            name, int(x.size), float(x.sum()), a_prior, b_prior, replace
        )
