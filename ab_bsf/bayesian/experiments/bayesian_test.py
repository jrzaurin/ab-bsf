import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ab_bsf.bayesian.metrics.evaluation import (
    estimate_credible_intervals,
    estimate_expected_loss,
    estimate_probabilities,
    estimate_relative_credible_intervals,
    estimate_relative_differences,
)
from ab_bsf.bayesian.models import ConjugateModel
from ab_bsf.bayesian.models.base import Data


class BayesianTest:
    """
    Bayesian A/B(/C...) test: compares the variants of one ``ConjugateModel``.

    Each variant is stored in ``self.data[name]`` as the sufficient statistics of
    its data. All variants share the model's prior. When appending with
    ``replace=False`` the statistics are summed.

    The first variant added is the reference for relative differences unless
    ``reference`` is passed to ``evaluate``.

    Priors are uninformative unless set in the model or with ``fit_priors``. With
    informative priors, ``prior_fraction`` in ``evaluate`` and the other result
    methods sets their weight at analysis time: ``0.2`` makes the prior worth 20%
    of the observations of the average variant, the same for every variant. With
    the default weak priors it would inflate them.

    Parameters
    ----------
    model : the conjugate model shared by all variants.
    """

    def __init__(self, model: ConjugateModel) -> None:
        self.model = model
        self.data: Dict[str, Dict[str, Any]] = {}

    @property
    def variant_names(self) -> List[str]:
        return list(self.data)

    def fit_priors(self, historical: Data) -> "BayesianTest":
        """Set the model's prior from historical data. Returns ``self``."""
        self.model.fit_priors(historical)
        return self

    def add_variant_data(self, name: str, data: Data, replace: bool = True) -> None:
        """Add the raw data of a variant. ``replace=False`` appends to it."""
        self._store(name, self.model.summarize(data), replace)

    def probabs_of_being_best(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
        prior_fraction: Optional[float] = None,
    ) -> Dict[str, float]:
        """Probability of each variant being best."""
        return self._by_variant(
            estimate_probabilities, sim_count, seed, prior_fraction, min_is_best
        )

    def expected_loss(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
        prior_fraction: Optional[float] = None,
    ) -> Dict[str, float]:
        """Expected loss of choosing each variant."""
        return self._by_variant(
            estimate_expected_loss, sim_count, seed, prior_fraction, min_is_best
        )

    def credible_intervals(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        interval_alpha: float = 0.95,
        prior_fraction: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """Credible interval of each variant's metric."""
        return self._by_variant(
            estimate_credible_intervals, sim_count, seed, prior_fraction, interval_alpha
        )

    def evaluate(
        self,
        sim_count: int = 20000,
        seed: Optional[int] = None,
        min_is_best: bool = False,
        interval_alpha: float = 0.95,
        reference: Optional[str] = None,
        return_samples: bool = True,
        prior_fraction: Optional[float] = None,
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
        prior_fraction : weight of the prior as a fraction of the observations of
            the average variant (default: use the prior as it is).

        Returns
        -------
        One dict per variant with its observed statistics, ``posterior_mean``,
        ``credible_interval``, ``relative_difference``, ``relative_credible_interval``,
        ``prob_being_best``, ``expected_loss`` and optionally ``posterior_samples``.
        """
        if not self.data:
            return []
        ref = self._reference_idx(reference)
        model = self._weighted_model(prior_fraction)
        samples = self._posterior_samples(model, sim_count, seed)
        statistics = list(self.data.values())

        observed = [self.model.observed(s) for s in statistics]
        columns: Dict[str, List[Any]] = {"variant": self.variant_names}
        columns.update({key: [row[key] for row in observed] for key in observed[0]})
        columns["posterior_mean"] = [
            round(model.posterior_mean(s), 5) for s in statistics
        ]
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

    def _store(self, name: str, statistics: Dict[str, Any], replace: bool) -> None:
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        self.model.validate(statistics)
        if name not in self.data:
            self.data[name] = statistics
        elif replace:
            warnings.warn(
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            self.data[name] = statistics
        else:
            warnings.warn(
                f"Variant {name} already exists - new data is appended to variant. "
                "If you wish to replace data instead, use replace=True."
            )
            for key, value in statistics.items():
                self.data[name][key] = self.data[name][key] + value

    def _weighted_model(self, prior_fraction: Optional[float]) -> ConjugateModel:
        if prior_fraction is None:
            return self.model
        if prior_fraction <= 0:
            raise ValueError("prior_fraction must be positive")
        mean_n = float(
            np.mean([self.model.n_observations(s) for s in self.data.values()])
        )
        return self.model.weighted(prior_fraction * mean_n)

    def _posterior_samples(
        self, model: ConjugateModel, sim_count: int, seed: Optional[int]
    ) -> np.ndarray:
        """Posterior draws of each variant's metric, shape (n_variants, sim_count)."""
        rng = np.random.default_rng(seed)
        return np.array(
            [model.posterior_samples(s, sim_count, rng) for s in self.data.values()]
        )

    def _by_variant(
        self,
        estimate,
        sim_count: int,
        seed: Optional[int],
        prior_fraction: Optional[float],
        *args,
    ) -> dict:
        if not self.data:
            return {}
        model = self._weighted_model(prior_fraction)
        samples = self._posterior_samples(model, sim_count, seed)
        return dict(zip(self.variant_names, estimate(samples, *args)))

    def _reference_idx(self, reference: Optional[str]) -> int:
        if reference is None:
            return 0
        if reference not in self.data:
            raise ValueError(
                f"Reference variant '{reference}' is not in the experiment."
            )
        return self.variant_names.index(reference)
