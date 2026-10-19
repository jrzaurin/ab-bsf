from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_normal_agg


class NormalBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for normal data with unknown mean and variance,
    Normal-Inverse-Gamma conjugate model. The compared metric is the mean ``mu``.

    - Likelihood: each observation is Normal(mu, sigma^2). The data enters only
      through ``totals``, ``sum_values`` and ``sum_values_2`` (sum of squares), which
      is why it can be passed raw or already aggregated.
    - Prior: ``sigma^2 ~ InverseGamma(a_prior_ig, b_prior_ig)`` and
      ``mu | sigma^2 ~ Normal(m_prior, sigma^2 / w_prior)``. ``m_prior`` is the prior
      mean and ``w_prior`` its weight in pseudo-observations (0.01 is very weak).
      The Inverse-Gamma parameters can be 0 because there is always at least one
      observation.
    - Posterior: same Normal-Inverse-Gamma form, with the data added to the prior.
      The posterior mean of ``mu`` is
      ``(sum_values + w_prior * m_prior) / (totals + w_prior)``, a weighted average
      of the prior mean and the sample mean.

    Conjugacy means the posterior stays in the prior's family, so the update has a
    closed form. The test draws ``mu`` from each variant's posterior and compares
    variants draw by draw to get the probability of being best, the expected loss and
    credible intervals reported by ``evaluate``.
    """

    @property
    def totals(self):
        return self._values("totals")

    @property
    def sum_values(self):
        return self._values("sum_values")

    @property
    def sum_values_2(self):
        return self._values("sum_values_2")

    @property
    def m_priors(self):
        return self._values("m_prior")

    @property
    def a_priors_ig(self):
        return self._values("a_prior_ig")

    @property
    def b_priors_ig(self):
        return self._values("b_prior_ig")

    @property
    def w_priors(self):
        return self._values("w_prior")

    def _posterior_samples(self, sim_count, seed):
        return sample_normal_agg(
            self.totals,
            self.sum_values,
            self.sum_values_2,
            self.m_priors,
            self.a_priors_ig,
            self.b_priors_ig,
            self.w_priors,
            sim_count,
            seed,
        )

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "sum_values": [round(s, 5) for s in self.sum_values],
            "avg_values": [
                round(s / t, 5) for s, t in zip(self.sum_values, self.totals)
            ],
        }

    def _posterior_means(self):
        return [
            round((s + w * m) / (t + w), 5)
            for s, t, m, w in zip(
                self.sum_values, self.totals, self.m_priors, self.w_priors
            )
        ]

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        sum_values: float,
        sum_values_2: float,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated normal data.

        Parameters
        ----------
        totals : number of observations.
        sum_values, sum_values_2 : sum and sum of squares of the observations.
        m_prior : prior mean.
        a_prior_ig, b_prior_ig : Inverse-Gamma prior on the variance (0 is allowed).
        w_prior : prior effective sample size (0.01 = very weak).
        """
        if totals <= 0:
            raise ValueError(
                "Input variable 'totals' is expected to be positive integer."
            )
        if a_prior_ig < 0 or b_prior_ig < 0 or w_prior < 0:
            raise ValueError(
                "All priors of [a_ig, b_ig, w] have to be non-negative numbers."
            )
        self._store(
            name,
            {"totals": totals, "sum_values": sum_values, "sum_values_2": sum_values_2},
            {
                "m_prior": m_prior,
                "a_prior_ig": a_prior_ig,
                "b_prior_ig": b_prior_ig,
                "w_prior": w_prior,
            },
            replace,
        )

    def add_variant_data(
        self,
        name: str,
        data: List[float],
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        replace: bool = True,
    ) -> None:
        """Add raw normal data."""
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        self.add_variant_data_agg(
            name,
            int(x.size),
            float(x.sum()),
            float((x**2).sum()),
            m_prior,
            a_prior_ig,
            b_prior_ig,
            w_prior,
            replace,
        )
