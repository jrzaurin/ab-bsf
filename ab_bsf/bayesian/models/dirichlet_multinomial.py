from typing import Any, Dict, Optional, Sequence

import numpy as np

from ab_bsf.bayesian.models.base import ConjugateModel, Data, Statistics

WEAK = 1


class DirichletMultinomial(ConjugateModel):
    """
    Dirichlet-Multinomial conjugate model for numeric categorical outcomes
    (dice, 1-5 star ratings). The compared metric is the expected value.

    This is the multi-category version of ``BetaBernoulli``: the Multinomial
    generalises the Binomial, and the Dirichlet generalises the Beta.

    - Likelihood: the count of each category follows a Multinomial(n, p), where
      ``p`` is the vector of category probabilities. The data enters only
      through the ``concentration`` (the count per category).
    - Prior: ``p ~ Dirichlet(prior)``, one pseudo-count per category (default 1
      each, i.e. uniform).
    - Posterior: ``p | data ~ Dirichlet(prior + concentration)``, with mean
      probabilities ``(prior_k + count_k) / sum_j (prior_j + count_j)``.

    The metric is the expected value of the outcome, ``sum_k category_k x p_k``.
    Variants are compared by drawing ``p`` from each posterior and converting
    every draw to its expected value.

    Parameters
    ----------
    categories : the possible numeric outcomes, e.g. [1, 2, 3, 4, 5]. prior :
    one pseudo-count per category (default: 1 each).
    """

    def __init__(
        self, categories: Sequence[float], prior: Optional[Sequence[float]] = None
    ) -> None:
        if len(categories) < 2:
            raise ValueError("Need at least two categories.")
        prior = [WEAK] * len(categories) if prior is None else prior
        if len(prior) != len(categories):
            raise ValueError("Prior needs one pseudo-count per category.")
        self.categories = list(categories)
        self.prior: Dict[str, Any] = {"prior": list(prior)}

    def summarize(self, data: Data) -> Statistics:
        x = np.asarray(data)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not np.isin(x, self.categories).all():
            raise ValueError("All data values must be one of the categories.")
        return {"concentration": np.array([(x == c).sum() for c in self.categories])}

    def validate(self, statistics: Statistics) -> None:
        if len(statistics["concentration"]) != len(self.categories):
            raise ValueError("Concentration needs one count per category.")

    def fit_priors(self, historical: Data) -> "DirichletMultinomial":
        """The counts of the history added to the default prior of 1 per category."""
        counts = self.summarize(historical)["concentration"]
        self.prior = {"prior": (counts + WEAK).tolist()}
        return self

    def weighted(self, weight: float) -> "DirichletMultinomial":
        prior = np.asarray(self.prior["prior"], dtype=float)
        return DirichletMultinomial(self.categories, prior * weight / prior.sum())

    def n_observations(self, statistics: Statistics) -> float:
        return float(np.sum(statistics["concentration"]))

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        counts = np.asarray(statistics["concentration"])
        return {
            "concentration": dict(zip(self.categories, counts.tolist())),
            "average_value": round(float(counts @ self.categories / counts.sum()), 5),
        }

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        alpha = np.add(self.prior["prior"], statistics["concentration"])
        return rng.dirichlet(alpha, sim_count) @ np.asarray(
            self.categories, dtype=float
        )

    def posterior_mean(self, statistics: Statistics) -> float:
        alpha = np.add(self.prior["prior"], statistics["concentration"])
        return float(alpha @ self.categories / alpha.sum())
