from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_numerical_dirichlet_agg


class DiscreteBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for numeric categorical outcomes (dice, 1-5 star ratings),
    Dirichlet-Multinomial model. The compared metric is the expected value.

    Parameters
    ----------
    categories : the possible numeric outcomes, e.g. [1, 2, 3, 4, 5].
    """

    _aggregate_keys = ("concentration",)

    def __init__(self, categories: List[float]) -> None:
        super().__init__()
        if len(categories) < 2:
            raise ValueError("Need at least two categories.")
        self.categories = list(categories)

    def _store(self, name, aggregates, priors, replace):
        if not replace and name in self.data:
            # concentrations are vectors, so append element-wise
            self.data[name]["concentration"] = [
                a + b for a, b in zip(self.data[name]["concentration"], aggregates["concentration"])
            ]
            return
        super()._store(name, aggregates, priors, replace)

    @property
    def concentrations(self):
        return self._values("concentration")

    def _posterior_samples(self, sim_count, seed):
        return sample_numerical_dirichlet_agg(self.categories, self.concentrations, self._values("prior"), sim_count, seed)

    def _observed_columns(self) -> Dict[str, List[Any]]:
        cats = np.asarray(self.categories, dtype=float)
        return {
            "concentration": [dict(zip(self.categories, c)) for c in self.concentrations],
            "average_value": [round(float(np.dot(c, cats) / np.sum(c)), 5) for c in self.concentrations],
        }

    def add_variant_data_agg(
        self, name: str, concentration: List[int], prior: List[float] = None, replace: bool = True
    ) -> None:
        """Add counts per category (same order as ``categories``). Default prior: 1 per category."""
        if len(concentration) != len(self.categories):
            raise ValueError("Concentration needs one count per category.")
        prior = prior or [1] * len(self.categories)
        self._store(name, {"concentration": list(concentration)}, {"prior": list(prior)}, replace)

    def add_variant_data(self, name: str, data: List[float], prior: List[float] = None, replace: bool = True) -> None:
        """Add raw outcomes; every value must be one of ``categories``."""
        if not all(x in self.categories for x in data):
            raise ValueError("All data values must be one of the categories.")
        self.add_variant_data_agg(name, [sum(1 for x in data if x == c) for c in self.categories], prior, replace)
