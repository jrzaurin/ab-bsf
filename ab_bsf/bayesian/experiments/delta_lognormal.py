from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import DeltaLognormal


class DeltaLognormalBayesianTest(BayesianTest):
    """
    Bayesian A/B test for zero-inflated, right-skewed data (revenue or volume per
    session), with the ``DeltaLognormal`` model (see it for the math). The compared
    metric is the mean per observation: P(non-zero) x E[value | non-zero].

    Parameters
    ----------
    model : model with the prior shared by all variants (default: weak). Its
        ``min_positive`` sets which values count as zeros.
    """

    def __init__(self, model: Optional[DeltaLognormal] = None) -> None:
        super().__init__(model or DeltaLognormal())

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        positives: int,
        sum_values: float,
        sum_logs: float,
        sum_logs_2: float,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated delta-lognormal data.

        Parameters
        ----------
        name : variant name.
        totals : number of observations (e.g. sessions).
        positives : number of non-zero observations.
        sum_values : sum of the non-zero values.
        sum_logs, sum_logs_2 : sum of log(value) and of log(value)^2 over non-zeros.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(
            name,
            {
                "totals": totals,
                "positives": positives,
                "sum_values": sum_values,
                "sum_logs": sum_logs,
                "sum_logs_2": sum_logs_2,
            },
            replace,
        )


DeltaLogNormalBayesianTest = DeltaLognormalBayesianTest
