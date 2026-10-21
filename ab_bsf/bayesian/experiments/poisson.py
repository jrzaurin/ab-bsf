from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import GammaPoisson


class PoissonBayesianTest(BayesianTest):
    """
    Bayesian A/B test for counts (trades per session, sessions per user), with the
    ``GammaPoisson`` conjugate model (see it for the math). The compared metric is
    the Poisson rate.

    Parameters
    ----------
    model : model with the prior shared by all variants (default: Gamma(0.1, 0.1)).
    """

    def __init__(self, model: Optional[GammaPoisson] = None) -> None:
        super().__init__(model or GammaPoisson())

    def add_variant_data_agg(
        self, name: str, totals: int, sum_values: float, replace: bool = True
    ) -> None:
        """
        Add aggregated counts.

        Parameters
        ----------
        name : variant name.
        totals : number of observations.
        sum_values : sum of the counts.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(name, {"totals": totals, "sum_values": sum_values}, replace)
