from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_numerical_dirichlet_agg


class DiscreteBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for numeric categorical outcomes (dice, 1-5 star ratings),
    Dirichlet-Multinomial conjugate model. The compared metric is the expected value.

    This is the multi-category version of ``BinaryBayesianTest``: the Multinomial
    generalises the Binomial, and the Dirichlet generalises the Beta.

    - Likelihood: the count of each category follows a Multinomial(n, p), where
      ``p`` is the vector of category probabilities. The data enters only through
      the ``concentration`` (the count per category).
    - Prior: ``p ~ Dirichlet(prior)``, one pseudo-count per category (default 1 each,
      i.e. uniform).
    - Posterior: ``p | data ~ Dirichlet(prior + concentration)``, with mean
      probabilities ``(prior_k + count_k) / sum_j (prior_j + count_j)``.

    The metric is the expected value of the outcome, ``sum_k category_k x p_k``.
    The test draws ``p`` from each variant's posterior, converts every draw to its
    expected value, and compares variants draw by draw to get the probability of
    being best, the expected loss and credible intervals reported by ``evaluate``.

    Parameters
    ----------
    categories : the possible numeric outcomes, e.g. [1, 2, 3, 4, 5].
    """

    def __init__(self, categories: List[float]) -> None:
        super().__init__()
        if len(categories) < 2:
            raise ValueError("Need at least two categories.")
        self.categories = list(categories)

    def _store(self, name, aggregates, priors, replace):
        if not replace and name in self.data:
            # concentrations are vectors, so append element-wise
            self.data[name]["concentration"] = [
                a + b
                for a, b in zip(
                    self.data[name]["concentration"], aggregates["concentration"]
                )
            ]
            return
        super()._store(name, aggregates, priors, replace)

    @property
    def concentrations(self):
        return self._values("concentration")

    def _posterior_samples(self, sim_count, seed):
        return sample_numerical_dirichlet_agg(
            self.categories, self.concentrations, self._values("prior"), sim_count, seed
        )

    def _observed_columns(self) -> Dict[str, List[Any]]:
        cats = np.asarray(self.categories, dtype=float)
        return {
            "concentration": [
                dict(zip(self.categories, c)) for c in self.concentrations
            ],
            "average_value": [
                round(float(np.dot(c, cats) / np.sum(c)), 5)
                for c in self.concentrations
            ],
        }

    def _posterior_means(self):
        # E[p_k] = (prior_k + count_k) / sum_j (prior_j + count_j), so the expected
        # value is the dot product of the categories with that mean probability
        cats = np.asarray(self.categories, dtype=float)
        return [
            round(float(np.dot(np.add(p, c), cats) / (np.sum(p) + np.sum(c))), 5)
            for p, c in zip(self._values("prior"), self.concentrations)
        ]

    def add_variant_data_agg(
        self,
        name: str,
        concentration: List[int],
        prior: List[float] = None,
        replace: bool = True,
    ) -> None:
        """Add counts per category (same order as ``categories``). Default prior: 1 per category."""
        if len(concentration) != len(self.categories):
            raise ValueError("Concentration needs one count per category.")
        prior = prior or [1] * len(self.categories)
        self._store(
            name,
            {"concentration": list(concentration)},
            {"prior": list(prior)},
            replace,
        )

    def add_variant_data(
        self,
        name: str,
        data: List[float],
        prior: List[float] = None,
        replace: bool = True,
    ) -> None:
        """Add raw outcomes; every value must be one of ``categories``."""
        if not all(x in self.categories for x in data):
            raise ValueError("All data values must be one of the categories.")
        self.add_variant_data_agg(
            name,
            [sum(1 for x in data if x == c) for c in self.categories],
            prior,
            replace,
        )
