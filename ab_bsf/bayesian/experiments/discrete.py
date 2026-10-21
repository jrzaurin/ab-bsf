from typing import List

import numpy as np

from ab_bsf.bayesian.experiments.bayesian_test import BayesianTest
from ab_bsf.bayesian.models import DirichletMultinomial


class DiscreteBayesianTest(BayesianTest):
    """
    Bayesian A/B test for numeric categorical outcomes (dice, 1-5 star ratings),
    with the ``DirichletMultinomial`` conjugate model (see it for the math). The
    compared metric is the expected value.

    Parameters
    ----------
    model : model with the categories and the prior shared by all variants, e.g.
        ``DirichletMultinomial([1, 2, 3, 4, 5])``.
    """

    def __init__(self, model: DirichletMultinomial) -> None:
        super().__init__(model)

    def add_variant_data_agg(
        self, name: str, concentration: List[int], replace: bool = True
    ) -> None:
        """
        Add the count of each category (same order as the model's categories).

        Parameters
        ----------
        name : variant name.
        concentration : count per category.
        replace : replace an existing variant (True) or append to it (False).
        """
        self._store(name, {"concentration": np.array(concentration)}, replace)
