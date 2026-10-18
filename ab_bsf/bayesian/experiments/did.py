"""
Bayesian difference-in-differences with aggregate pre/post data.

DiD = (treatment_post - treatment_pre) - (control_post - control_pre), computed
draw by draw from four independent posteriors. Appropriate when the pre and post
periods are separate samples (e.g. different sessions); with the same users in
both periods use ``NormalBayesianTest.add_variant_data_did`` on per-user changes.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ab_bsf.bayesian.metrics.evaluation import sample_bernoulli_agg

_PERIODS = ("pre", "post")


class BaseDidTest(ABC):
    """Groups x periods container; subclasses define how a cell is sampled."""

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def group_names(self) -> List[str]:
        return list(self.data)

    def _check(self, *groups: str) -> None:
        for g in groups:
            if g not in self.data:
                raise ValueError(f"Group '{g}' not found")
            for p in _PERIODS:
                if p not in self.data[g]:
                    raise ValueError(f"Group '{g}' has no '{p}' data")

    @abstractmethod
    def _cell_samples(self, cells: List[Dict[str, Any]], sim_count: int, seed) -> np.ndarray:
        """Posterior draws of the mean metric for each cell, shape (n_cells, sim_count)."""

    def evaluate(
        self,
        treatment_group: str,
        control_group: str,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        interval_alpha: float = 0.95,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with ``did`` (posterior mean), ``credible_interval``, ``prob_positive``,
        ``expected_loss_if_ship`` (E[max(0, -DiD)]), ``relative_did`` (DiD over the
        treatment's counterfactual post level), the posterior mean of every cell,
        and optionally ``did_samples``.
        """
        self._check(treatment_group, control_group)
        cells = [self.data[g][p] for g in (treatment_group, control_group) for p in _PERIODS]
        t_pre, t_post, c_pre, c_post = self._cell_samples(cells, sim_count, seed)
        did = (t_post - t_pre) - (c_post - c_pre)
        counterfactual = t_pre + (c_post - c_pre)
        lo, hi = np.quantile(did, [(1 - interval_alpha) / 2, (1 + interval_alpha) / 2])
        out = {
            "did": float(did.mean()),
            "credible_interval": [float(lo), float(hi)],
            "prob_positive": float((did > 0).mean()),
            "expected_loss_if_ship": float(np.maximum(0, -did).mean()),
            "relative_did": float(np.mean(did / np.where(counterfactual == 0, np.nan, counterfactual))),
            "means": {
                f"{treatment_group}_pre": float(t_pre.mean()),
                f"{treatment_group}_post": float(t_post.mean()),
                f"{control_group}_pre": float(c_pre.mean()),
                f"{control_group}_post": float(c_post.mean()),
            },
        }
        if return_samples:
            out["did_samples"] = did
        return out

    def delete_group(self, name: str) -> None:
        if name not in self.data:
            warnings.warn(f"Nothing to be deleted. Group {name} is not in experiment.")
        else:
            del self.data[name]


class BinaryDidTest(BaseDidTest):
    """
    DiD for conversion rates from aggregate counts (four Beta posteriors).

    Examples
    --------
    >>> t = BinaryDidTest()
    >>> t.add_group_data("treatment", "pre", totals=1000, positives=120)
    >>> t.add_group_data("treatment", "post", totals=1000, positives=170)
    >>> t.add_group_data("control", "pre", totals=1000, positives=100)
    >>> t.add_group_data("control", "post", totals=1000, positives=110)
    >>> round(t.evaluate("treatment", "control", seed=0)["did"], 2)
    0.04
    """

    def add_group_data(
        self, group_name: str, period: str, totals: int, positives: int, a_prior: float = 0.5, b_prior: float = 0.5
    ) -> None:
        if period not in _PERIODS:
            raise ValueError("Period must be either 'pre' or 'post'")
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] must be positive")
        if totals <= 0 or positives < 0 or positives > totals:
            raise ValueError("Need totals > 0 and 0 <= positives <= totals")
        self.data.setdefault(group_name, {})[period] = {
            "totals": totals,
            "positives": positives,
            "a_prior": a_prior,
            "b_prior": b_prior,
        }

    def _cell_samples(self, cells, sim_count, seed):
        return sample_bernoulli_agg(
            [c["totals"] for c in cells],
            [c["positives"] for c in cells],
            [c["a_prior"] for c in cells],
            [c["b_prior"] for c in cells],
            sim_count,
            seed,
        )
