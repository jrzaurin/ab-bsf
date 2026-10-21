from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import GammaExponential


class ExponentialBayesianTest(BayesianTest):
    """
    Bayesian A/B test for positive durations (session time, time to first trade),
    with the ``GammaExponential`` conjugate model (see it for the math). The
    compared metric is the mean duration; use ``evaluate(min_is_best=True)`` when
    shorter is better.

    Parameters
    ----------
    model : model with the prior shared by all variants (default: Gamma(0.1, 0.1)
        on the rate).
    """

    def __init__(self, model: Optional[GammaExponential] = None) -> None:
        super().__init__(model or GammaExponential())

    def add_variant_data_agg(
        self, name: str, totals: int, sum_values: float, replace: bool = True
    ) -> None:
        """
        Add aggregated durations.

        Parameters
        ----------
        name : variant name.
        totals : number of observations.
        sum_values : sum of the durations.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(name, {"totals": totals, "sum_values": sum_values}, replace)
