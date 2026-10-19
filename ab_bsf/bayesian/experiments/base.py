"""
The code in this module is based on the code in the following repository:
https://github.com/Matt52/bayesian-testing

Therefore, all credits to the original author.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ab_bsf.bayesian.metrics.evaluation import (
    estimate_credible_intervals,
    estimate_expected_loss,
    estimate_probabilities,
    estimate_relative_credible_intervals,
    estimate_relative_differences,
)


class BaseBayesianTest(ABC):
    """
    Base class for Bayesian A/B(/C...) tests.

    Each variant is stored in ``self.data[name]`` as a dict of aggregated
    sufficient statistics plus its prior parameters. When appending with
    ``replace=False`` the aggregates are summed and the original priors are
    kept. Subclasses implement ``_posterior_samples``, ``_posterior_means`` and
    ``_observed_columns``.

    The first variant added is the reference for relative differences unless
    ``reference`` is passed to ``evaluate``.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    @property
    def variant_names(self) -> List[str]:
        return list(self.data)

    def probabs_of_being_best(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
    ) -> Dict[str, float]:
        """Probability of each variant being best."""
        return self._by_variant(estimate_probabilities, sim_count, seed, min_is_best)

    def expected_loss(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
    ) -> Dict[str, float]:
        """Expected loss of choosing each variant."""
        return self._by_variant(estimate_expected_loss, sim_count, seed, min_is_best)

    def credible_intervals(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        interval_alpha: float = 0.95,
    ) -> Dict[str, List[float]]:
        """Credible interval of each variant's metric."""
        return self._by_variant(
            estimate_credible_intervals, sim_count, seed, interval_alpha
        )

    def evaluate(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
        interval_alpha: float = 0.95,
        reference: Optional[str] = None,
        return_samples: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the experiment.

        Parameters
        ----------
        sim_count : number of posterior draws.
        seed : random seed.
        min_is_best : treat the smallest metric as best (e.g. waiting time).
        interval_alpha : probability of the credible intervals.
        reference : variant used for relative differences (default: first added).
        return_samples : include each variant's posterior draws under ``posterior_samples``.

        Returns
        -------
        One dict per variant with its observed statistics, ``posterior_mean``,
        ``credible_interval``, ``relative_difference``, ``relative_credible_interval``,
        ``prob_being_best``, ``expected_loss`` and optionally ``posterior_samples``.
        """
        if not self.data:
            return []
        ref = self._reference_idx(reference)
        samples = self._posterior_samples(sim_count, seed)

        columns: Dict[str, List[Any]] = {"variant": self.variant_names}
        columns.update(self._observed_columns())
        columns["posterior_mean"] = self._posterior_means()
        columns["credible_interval"] = estimate_credible_intervals(
            samples, interval_alpha
        )
        columns["relative_difference"] = estimate_relative_differences(samples, ref)
        columns["relative_credible_interval"] = estimate_relative_credible_intervals(
            samples, ref, interval_alpha
        )
        columns["prob_being_best"] = estimate_probabilities(samples, min_is_best)
        columns["expected_loss"] = estimate_expected_loss(samples, min_is_best)
        if return_samples:
            columns["posterior_samples"] = list(samples)

        keys = list(columns)
        return [dict(zip(keys, row)) for row in zip(*columns.values())]

    def delete_variant(self, name: str) -> None:
        """Delete a variant and all its data."""
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if name not in self.data:
            warnings.warn(
                f"Nothing to be deleted. Variant {name} is not in experiment."
            )
        else:
            del self.data[name]

    def _store(
        self,
        name: str,
        aggregates: Dict[str, Any],
        priors: Dict[str, Any],
        replace: bool,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if name not in self.data:
            self.data[name] = {**aggregates, **priors}
        elif replace:
            warnings.warn(
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            self.data[name] = {**aggregates, **priors}
        else:
            warnings.warn(
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. If you wish to replace data instead, use replace=True."
            )
            for key, value in aggregates.items():
                self.data[name][key] += value

    def _values(self, key: str) -> List[Any]:
        return [self.data[k][key] for k in self.data]

    def _by_variant(self, estimate, sim_count: int, seed: Optional[int], *args) -> dict:
        if not self.data:
            return {}
        samples = self._posterior_samples(sim_count, seed)
        return dict(zip(self.variant_names, estimate(samples, *args)))

    def _reference_idx(self, reference: Optional[str]) -> int:
        if reference is None:
            return 0
        if reference not in self.data:
            raise ValueError(
                f"Reference variant '{reference}' is not in the experiment."
            )
        return self.variant_names.index(reference)

    @abstractmethod
    def _posterior_samples(self, sim_count: int, seed: Optional[int]) -> np.ndarray:
        """Posterior draws of each variant's mean metric, shape (n_variants, sim_count)."""

    @abstractmethod
    def _observed_columns(self) -> Dict[str, List[Any]]:
        """Observed (data) columns reported by ``evaluate``, in display order."""

    @abstractmethod
    def _posterior_means(self) -> List[float]:
        """Posterior mean of the metric, in closed form from the conjugate update."""
