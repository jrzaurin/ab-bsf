from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_delta_normal_agg


class DeltaNormalBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for zero-inflated data whose non-zero values can be positive
    or negative (P&L, net deposits), with two conjugate models combined.

    Like ``DeltaLognormalBayesianTest``, each variant's data is split in two parts,
    modelled independently, but the non-zero values are modelled on their own scale
    instead of in log space:

    1. Whether an observation is non-zero. ``non_zeros`` out of ``totals`` is
       Binomial(totals, p), the prior is ``p ~ Beta(a_prior_beta, b_prior_beta)``
       and the posterior is
       ``Beta(a_prior_beta + non_zeros, b_prior_beta + totals - non_zeros)``.
    2. The size of the non-zero values, assumed Normal(mu, sigma^2) with both
       parameters unknown. The conjugate prior is Normal-Inverse-Gamma:
       ``sigma^2 ~ InverseGamma(a_prior_ig, b_prior_ig)`` and
       ``mu | sigma^2 ~ Normal(m_prior, sigma^2 / w_prior)``, where ``w_prior`` is
       the prior's weight in pseudo-observations. The posterior has the same form.

    The compared metric is the mean per observation, ``p x mu``. The test draws from
    both posteriors, multiplies the draws, and compares variants draw by draw to get
    the probability of being best, the expected loss and credible intervals reported
    by ``evaluate``. The reported ``posterior_mean`` is exact: E[p] x E[mu], since
    both parts are independent.
    """

    @property
    def totals(self):
        return self._values("totals")

    @property
    def non_zeros(self):
        return self._values("non_zeros")

    @property
    def sum_values(self):
        return self._values("sum_values")

    @property
    def sum_values_2(self):
        return self._values("sum_values_2")

    def _posterior_samples(self, sim_count, seed):
        return sample_delta_normal_agg(
            self.totals,
            self.non_zeros,
            self.sum_values,
            self.sum_values_2,
            self._values("a_prior_beta"),
            self._values("b_prior_beta"),
            self._values("m_prior"),
            self._values("a_prior_ig"),
            self._values("b_prior_ig"),
            self._values("w_prior"),
            sim_count,
            seed,
        )

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "non_zeros": self.non_zeros,
            "sum_values": [round(s, 5) for s in self.sum_values],
            "avg_values": [
                round(s / t, 5) for s, t in zip(self.sum_values, self.totals)
            ],
            "avg_non_zero_values": [
                round(s / n, 5) if n else 0.0
                for s, n in zip(self.sum_values, self.non_zeros)
            ],
        }

    def _posterior_means(self) -> List[float]:
        """
        Exact posterior mean from conjugacy, no sampling involved:
        E[P(non-zero)] x E[mu] = (a + n) / (a + b + totals) x (sum + w * m) / (n + w).
        """
        means = []
        rows = zip(
            self.totals,
            self.non_zeros,
            self.sum_values,
            self._values("a_prior_beta"),
            self._values("b_prior_beta"),
            self._values("m_prior"),
            self._values("w_prior"),
        )
        for total, n, s, a_beta, b_beta, m, w in rows:
            if n == 0:
                means.append(0.0)
                continue
            p_non_zero = (a_beta + n) / (a_beta + b_beta + total)
            means.append(round(p_non_zero * (s + w * m) / (n + w), 5))
        return means

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        non_zeros: int,
        sum_values: float,
        sum_values_2: float,
        a_prior_beta: float = 0.5,
        b_prior_beta: float = 0.5,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        replace: bool = True,
    ) -> None:
        """Add aggregated delta-normal data (sums are over the non-zero values)."""
        if a_prior_beta <= 0 or b_prior_beta <= 0:
            raise ValueError(
                "Both [a_prior_beta, b_prior_beta] have to be positive numbers."
            )
        if totals <= 0 or non_zeros < 0 or totals < non_zeros:
            raise ValueError("Need totals > 0 and 0 <= non_zeros <= totals.")
        self._store(
            name,
            {
                "totals": totals,
                "non_zeros": non_zeros,
                "sum_values": sum_values,
                "sum_values_2": sum_values_2,
            },
            {
                "a_prior_beta": a_prior_beta,
                "b_prior_beta": b_prior_beta,
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
        a_prior_beta: float = 0.5,
        b_prior_beta: float = 0.5,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        replace: bool = True,
    ) -> None:
        """Add raw data; zeros are the 'no event' mass."""
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        nz = x[x != 0]
        self.add_variant_data_agg(
            name,
            int(x.size),
            int(nz.size),
            float(nz.sum()),
            float((nz**2).sum()),
            a_prior_beta,
            b_prior_beta,
            m_prior,
            a_prior_ig,
            b_prior_ig,
            w_prior,
            replace,
        )
