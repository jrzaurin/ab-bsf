from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import DeltaNormal


class DeltaNormalBayesianTest(BayesianTest):
    """
    Bayesian A/B test for zero-inflated data whose non-zero values can be positive
    or negative (P&L, net deposits), with the ``DeltaNormal`` model (see it for the
    math). The compared metric is the mean per observation.

    Parameters
    ----------
    model : model with the prior shared by all variants (default: weak).
    """

    def __init__(self, model: Optional[DeltaNormal] = None) -> None:
        super().__init__(model or DeltaNormal())

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        non_zeros: int,
        sum_values: float,
        sum_values_2: float,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated delta-normal data.

        Parameters
        ----------
        name : variant name.
        totals : number of observations.
        non_zeros : number of non-zero observations.
        sum_values, sum_values_2 : sum and sum of squares of the non-zero values.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(
            name,
            {
                "totals": totals,
                "non_zeros": non_zeros,
                "sum_values": sum_values,
                "sum_values_2": sum_values_2,
            },
            replace,
        )
