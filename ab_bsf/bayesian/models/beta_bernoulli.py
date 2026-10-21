from typing import Any, Dict

import numpy as np

from ab_bsf.bayesian.models.base import ConjugateModel, Data, Statistics

# uninformative prior
JEFFREYS = 0.5


class BetaBernoulli(ConjugateModel):
    """
    Beta-Binomial conjugate model for binary data (conversions).

    - Likelihood: each observation is a Bernoulli(p) trial, so ``positives`` out
      of ``totals`` follows a Binomial(totals, p).
    - Prior: ``p ~ Beta(a_prior, b_prior)``. Read ``a_prior`` and ``b_prior`` as
      pseudo-conversions and pseudo-non-conversions seen before the experiment.
      The default is Beta(0.5, 0.5) (Jeffreys).
    - Posterior: ``p | data ~ Beta(a_prior + positives, b_prior + totals -
      positives)``.
    """

    def __init__(self, a_prior: float = JEFFREYS, b_prior: float = JEFFREYS) -> None:
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        self.prior: Dict[str, Any] = {"a_prior": a_prior, "b_prior": b_prior}

    def summarize(self, data: Data) -> Statistics:
        x = np.asarray(data)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not np.isin(x, (0, 1)).all():
            raise ValueError("Input data needs to be a list of zeros and ones.")
        return {"totals": int(x.size), "positives": int(x.sum())}

    def validate(self, statistics: Statistics) -> None:
        totals, positives = statistics["totals"], statistics["positives"]
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

    def fit_priors(self, historical: Data) -> "BetaBernoulli":
        """The posterior of the history under the default Beta(0.5, 0.5) prior."""
        statistics = self.summarize(historical)
        positives = statistics["positives"]
        self.prior = {
            "a_prior": positives + JEFFREYS,
            "b_prior": statistics["totals"] - positives + JEFFREYS,
        }
        return self

    def weighted(self, weight: float) -> "BetaBernoulli":
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        scale = weight / (a + b)
        return BetaBernoulli(a * scale, b * scale)

    def n_observations(self, statistics: Statistics) -> float:
        return statistics["totals"]

    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        totals, positives = statistics["totals"], statistics["positives"]
        return {
            "totals": totals,
            "positives": positives,
            "positive_rate": round(positives / totals, 5),
        }

    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        totals, positives = statistics["totals"], statistics["positives"]
        return rng.beta(
            self.prior["a_prior"] + positives,
            self.prior["b_prior"] + totals - positives,
            sim_count,
        )

    def posterior_mean(self, statistics: Statistics) -> float:
        totals, positives = statistics["totals"], statistics["positives"]
        a, b = self.prior["a_prior"], self.prior["b_prior"]
        return (a + positives) / (a + b + totals)
