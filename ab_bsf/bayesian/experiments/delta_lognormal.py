from typing import Any, Dict, List

import numpy as np

from ab_bsf.bayesian.experiments.base import BaseBayesianTest
from ab_bsf.bayesian.metrics.evaluation import sample_delta_lognormal_agg


class DeltaLognormalBayesianTest(BaseBayesianTest):
    """
    Bayesian A/B test for zero-inflated, right-skewed data (revenue or volume per
    session): P(non-zero) ~ Beta-Bernoulli, non-zero values ~ LogNormal.

    The compared metric is the mean per observation, P(non-zero) x E[value | non-zero].
    """

    _aggregate_keys = ("totals", "positives", "sum_values", "sum_logs", "sum_logs_2")

    @property
    def totals(self):
        return self._values("totals")

    @property
    def positives(self):
        return self._values("positives")

    @property
    def sum_values(self):
        return self._values("sum_values")

    @property
    def sum_logs(self):
        return self._values("sum_logs")

    @property
    def sum_logs_2(self):
        return self._values("sum_logs_2")

    def _posterior_samples(self, sim_count, seed):
        v = self._values
        return sample_delta_lognormal_agg(
            self.totals,
            self.positives,
            self.sum_logs,
            self.sum_logs_2,
            v("a_prior_beta"),
            v("b_prior_beta"),
            v("m_prior"),
            v("a_prior_ig"),
            v("b_prior_ig"),
            v("w_prior"),
            sim_count,
            seed,
        )

    def _observed_columns(self) -> Dict[str, List[Any]]:
        return {
            "totals": self.totals,
            "positives": self.positives,
            "sum_values": [round(s, 5) for s in self.sum_values],
            "avg_values": [round(s / t, 5) for s, t in zip(self.sum_values, self.totals)],
            "avg_positive_values": [round(s / p, 5) if p else 0.0 for s, p in zip(self.sum_values, self.positives)],
        }

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        positives: int,
        sum_values: float,
        sum_logs: float,
        sum_logs_2: float,
        a_prior_beta: float = 0.5,
        b_prior_beta: float = 0.5,
        m_prior: float = 1,
        a_prior_ig: float = 0,
        b_prior_ig: float = 0,
        w_prior: float = 0.01,
        replace: bool = True,
    ) -> None:
        """
        Add aggregated delta-lognormal data.

        Parameters
        ----------
        totals : number of observations (e.g. sessions).
        positives : number of non-zero observations.
        sum_values : sum of the non-zero values.
        sum_logs, sum_logs_2 : sum of log(value) and of log(value)^2 over non-zeros.
        a_prior_beta, b_prior_beta : Beta prior on P(non-zero).
        m_prior : prior mean of log(value) (log space: m_prior=1 -> geometric mean e).
        a_prior_ig, b_prior_ig, w_prior : NIG prior on the log values.
        """
        if a_prior_beta <= 0 or b_prior_beta <= 0:
            raise ValueError("Both [a_prior_beta, b_prior_beta] have to be positive numbers.")
        if a_prior_ig < 0 or b_prior_ig < 0 or w_prior < 0:
            raise ValueError("All priors of [a_ig, b_ig, w] have to be non-negative numbers.")
        if totals <= 0:
            raise ValueError("Input variable 'totals' is expected to be positive integer.")
        if positives < 0:
            raise ValueError("Input variable 'positives' is expected to be a non-negative integer.")
        if totals < positives:
            raise ValueError("Not possible to have more positives that totals!")
        self._store(
            name,
            {
                "totals": totals,
                "positives": positives,
                "sum_values": sum_values,
                "sum_logs": sum_logs,
                "sum_logs_2": sum_logs_2,
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
        min_positive: float = 0.0,
    ) -> None:
        """
        Add raw data (zeros allowed, no negatives).

        ``min_positive`` treats values <= min_positive as zeros, e.g. to drop dust
        trades that would otherwise create a second mode in log space.
        """
        x = np.asarray(data, dtype=float)
        if x.size == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if (x < 0).any():
            raise ValueError("Input data needs to be a list of non-negative numbers.")
        pos = x[x > min_positive]
        logs = np.log(pos)
        self.add_variant_data_agg(
            name,
            int(x.size),
            int(pos.size),
            float(pos.sum()),
            float(logs.sum()),
            float((logs**2).sum()),
            a_prior_beta,
            b_prior_beta,
            m_prior,
            a_prior_ig,
            b_prior_ig,
            w_prior,
            replace,
        )


# Spelling used in some of the original scripts
DeltaLogNormalBayesianTest = DeltaLognormalBayesianTest
