from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import NormalInverseGamma


class NormalBayesianTest(BayesianTest):
    """
    Bayesian A/B test for normal data with unknown mean and variance, with the
    ``NormalInverseGamma`` conjugate model (see it for the math). The compared
    metric is the mean.

    Parameters
    ----------
    model : model with the prior shared by all variants (default: very weak).
    """

    def __init__(self, model: Optional[NormalInverseGamma] = None) -> None:
        super().__init__(model or NormalInverseGamma())

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        sum_values: float,
        sum_values_2: float,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated normal data.

        Parameters
        ----------
        name : variant name.
        totals : number of observations.
        sum_values, sum_values_2 : sum and sum of squares of the observations.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(
            name,
            {"totals": totals, "sum_values": sum_values, "sum_values_2": sum_values_2},
            replace,
        )
