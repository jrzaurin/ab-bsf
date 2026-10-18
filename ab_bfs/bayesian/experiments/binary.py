from typing import Any, Dict, List, Optional

import numpy as np

from ab_bfs.bayesian.experiments.base import BaseBayesianTest
from ab_bfs.bayesian.metrics.evaluation import (
    estimate_credible_intervals,
    estimate_expected_loss,
    estimate_probabilities,
    sample_bernoulli_agg,
)


class BinaryBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for binary data (conversions), Beta-Bernoulli conjugate model.

    Examples
    --------
    >>> test = BinaryBayesianTest()
    >>> test.add_variant_data_agg("control", totals=1000, positives=110)
    >>> test.add_variant_data_agg("treatment", totals=1000, positives=150)
    >>> res = test.evaluate(seed=1)
    >>> res[1]["prob_being_best"] > 0.99
    True
    """

    _aggregate_keys = ("totals", "positives")

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

    def _posterior_samples(self, sim_count, seed):
        return sample_bernoulli_agg(self.totals, self.positives, self.a_priors, self.b_priors, sim_count, seed)

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "positives": self.positives,
            "positive_rate": [round(p / t, 5) for p, t in zip(self.positives, self.totals)],
        }

    def _posterior_means(self, samples):
        return [
            round((a + p) / (a + b + t), 5) for p, t, a, b in zip(self.positives, self.totals, self.a_priors, self.b_priors)
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
            raise ValueError("Input variable 'totals' is expected to be positive integer.")
        if positives < 0:
            raise ValueError("Input variable 'positives' is expected to be non-negative integer.")
        if totals < positives:
            raise ValueError("Not possible to have more positives that totals!")
        self._store(name, {"totals": totals, "positives": positives}, {"a_prior": a_prior, "b_prior": b_prior}, replace)

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
        self.add_variant_data_agg(name, len(data), int(sum(data)), a_prior, b_prior, replace)

    def evaluate_with_did(
        self,
        pre_rates: Dict[str, float],
        sim_count: int = 20000,
        seed: Optional[int] = None,
        interval_alpha: float = 0.95,
        reference: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Difference-in-differences evaluation against known pre-period rates.

        Each variant's posterior conversion draws are shifted by its pre-period
        rate, so every metric refers to the *change* (post - pre). ``did`` is each
        variant's change minus the reference's change.

        Parameters
        ----------
        pre_rates : {variant: pre-period conversion rate}, treated as fixed.
        reference : variant to difference against (default: first added).

        Notes
        -----
        The pre-period rates are treated as known constants, which ignores their
        sampling noise and overstates certainty when the pre period is small. If you
        have pre-period counts, use :class:`BinaryDidTest`, which samples all four
        posteriors.
        """
        missing = [v for v in self.variant_names if v not in pre_rates]
        if missing:
            raise ValueError(f"Missing pre_rate for variant(s): {missing}")
        ref = self._reference_idx(reference)

        post = self._posterior_samples(sim_count, seed)
        change = post - np.array([[pre_rates[v]] for v in self.variant_names])
        did = change - change[ref]

        pbbs = estimate_probabilities(change)
        loss = estimate_expected_loss(change)
        ci_change = estimate_credible_intervals(change, interval_alpha)
        ci_did = estimate_credible_intervals(did, interval_alpha)

        out = []
        for i, v in enumerate(self.variant_names):
            out.append(
                {
                    "variant": v,
                    "totals": self.totals[i],
                    "positives": self.positives[i],
                    "positive_rate": round(self.positives[i] / self.totals[i], 5),
                    "pre_rate": pre_rates[v],
                    "change": round(float(change[i].mean()), 7),
                    "credible_interval": ci_change[i],
                    "did": 0.0 if i == ref else round(float(did[i].mean()), 7),
                    "did_credible_interval": [0.0, 0.0] if i == ref else ci_did[i],
                    "prob_did_positive": float("nan") if i == ref else round(float((did[i] > 0).mean()), 7),
                    "prob_being_best": pbbs[i],
                    "expected_loss": loss[i],
                    "posterior_samples": change[i],
                }
            )
        return out
