from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_exponential_agg


class ExponentialBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for positive durations (session time, time to first trade),
    Gamma-Exponential conjugate model. The compared metric is the mean, 1 / rate.
    Use ``evaluate(min_is_best=True)`` when shorter is better.
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
        return sample_exponential_agg(
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

    def _posterior_means(self, samples):
        # E[1/lambda] for lambda ~ Gamma(a + n, b + sum) is (b + sum) / (a + n - 1)
        return [
            (
                round((b + s) / (a + t - 1), 5)
                if a + t > 1
                else float(np.round(samples[i].mean(), 5))
            )
            for i, (s, t, a, b) in enumerate(
                zip(self.sum_values, self.totals, self.a_priors, self.b_priors)
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
        """Add aggregated data. Default prior Gamma(0.1, 0.1) on the rate (vague)."""
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0:
            raise ValueError(
                "Input variable 'totals' is expected to be positive integer."
            )
        if sum_values < 0:
            raise ValueError(
                "Input variable 'sum_values' is expected to be non-negative number."
            )
        self._store(
            name,
            {"totals": totals, "sum_values": sum_values},
            {"a_prior": a_prior, "b_prior": b_prior},
            replace,
        )

    def add_variant_data(
        self,
        name: str,
        data: List[float],
        a_prior: float = 0.1,
        b_prior: float = 0.1,
        replace: bool = True,
    ) -> None:
        """Add raw non-negative data."""
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if (x < 0).any():
            raise ValueError("Input data needs to be a list of non-negative numbers.")
        self.add_variant_data_agg(
            name, int(x.size), float(x.sum()), a_prior, b_prior, replace
        )
