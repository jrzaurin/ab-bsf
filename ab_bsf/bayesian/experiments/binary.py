from typing import Optional

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import BetaBernoulli


class BinaryBayesianTest(BayesianTest):
    """
    Bayesian A/B test for binary data (conversions), with the ``BetaBernoulli``
    conjugate model (see it for the math).

    The test draws samples from each variant's posterior and compares them draw by
    draw, which gives the probability of being best, the expected loss and credible
    intervals reported by ``evaluate``.

    Parameters
    ----------
    model : model with the prior shared by all variants (default: Beta(0.5, 0.5)).

    Examples
    --------
    >>> test = BinaryBayesianTest()
    >>> test.add_variant_data_agg("control", totals=1000, positives=110)
    >>> test.add_variant_data_agg("treatment", totals=1000, positives=150)
    >>> res = test.evaluate(seed=1)
    >>> res[1]["prob_being_best"] > 0.99
    True
    """

    def __init__(self, model: Optional[BetaBernoulli] = None) -> None:
        super().__init__(model or BetaBernoulli())

    def add_variant_data_agg(
        self, name: str, totals: int, positives: int, replace: bool = True
    ) -> None:
        """
        Add aggregated binary data.

        Parameters
        ----------
        name : variant name.
        totals : number of observations (e.g. sessions).
        positives : number of ones (e.g. conversions).
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(name, {"totals": totals, "positives": positives}, replace)
