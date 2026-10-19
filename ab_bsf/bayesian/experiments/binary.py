from typing import Any, Dict, List, Optional

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_bernoulli_agg


class BinaryBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for binary data (conversions), Beta-Binomial conjugate model.

    Each variant has an unknown conversion rate ``p``. A Bayesian analysis starts
    from a prior belief about ``p``, updates it with the observed data through the
    likelihood, and ends with a posterior: a full distribution over ``p``.

    A prior is *conjugate* to a likelihood when the posterior belongs to the same
    family as the prior, so the update has a closed form. Here:

    - Likelihood: each observation is a Bernoulli(p) trial, so ``positives`` out of
      ``totals`` follows a Binomial(totals, p).
    - Prior: ``p ~ Beta(a_prior, b_prior)``. Read ``a_prior`` and ``b_prior`` as
      pseudo-conversions and pseudo-non-conversions seen before the experiment.
    - Posterior: ``p | data ~ Beta(a_prior + positives, b_prior + totals - positives)``.

    The test draws samples from each variant's posterior and compares them draw by
    draw, which gives the probability of being best, the expected loss and credible
    intervals reported by ``evaluate``.

    Examples
    --------
    >>> test = BinaryBayesianTest()
    >>> test.add_variant_data_agg("control", totals=1000, positives=110)
    >>> test.add_variant_data_agg("treatment", totals=1000, positives=150)
    >>> res = test.evaluate(seed=1)
    >>> res[1]["prob_being_best"] > 0.99
    True
    """

    @property
    def totals(self):
        return self._values("totals")

    @property
    def positives(self):
        return self._values("positives")

    @property
    def a_priors(self):
        return self._values("a_prior")

    @property
    def b_priors(self):
        return self._values("b_prior")

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "positives": self.positives,
            "positive_rate": [
                round(p / t, 5) for p, t in zip(self.positives, self.totals)
            ],
        }

    def _posterior_samples(self, sim_count: int, seed: Optional[int]) -> np.ndarray:
        return sample_bernoulli_agg(
            self.totals, self.positives, self.a_priors, self.b_priors, sim_count, seed
        )

    def _posterior_means(self) -> List[float]:
        return [
            round((a + p) / (a + b + t), 5)
            for p, t, a, b in zip(
                self.positives, self.totals, self.a_priors, self.b_priors
            )
        ]

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        positives: int,
        a_prior: float = 0.5,
        b_prior: float = 0.5,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated binary data. Default prior Beta(0.5, 0.5) (Jeffreys).

        Parameters
        ----------
        name : variant name.
        totals : number of observations (e.g. sessions).
        positives : number of ones (e.g. conversions).
        a_prior, b_prior : Beta prior parameters.
        replace : replace an existing variant (True) or append to it (False).
        """
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0:
            raise ValueError(
                "Input variable 'totals' is expected to be positive integer."
            )
        if positives < 0:
            raise ValueError(
                "Input variable 'positives' is expected to be non-negative integer."
            )
        if totals < positives:
            raise ValueError("Not possible to have more positives that totals!")
        self._store(
            name,
            {"totals": totals, "positives": positives},
            {"a_prior": a_prior, "b_prior": b_prior},
            replace,
        )

    def add_variant_data(
        self,
        name: str,
        data: List[int],
        a_prior: float = 0.5,
        b_prior: float = 0.5,
        replace: bool = True,
    ) -> None:
        """Add raw 0/1 data."""
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not all(i in (0, 1) for i in data):
            raise ValueError("Input data needs to be a list of zeros and ones.")
        self.add_variant_data_agg(
            name, len(data), int(sum(data)), a_prior, b_prior, replace
        )
