from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Union

import numpy as np

Statistics = Dict[str, Any]
Data = Union[Sequence[float], np.ndarray]


class ConjugateModel(ABC):
    """
    A likelihood with its conjugate prior, for one variant.

    A model holds the prior in ``prior`` (a dict, uninformative unless set by the
    user or ``fit_priors``) and knows how to turn the data of a variant into its
    sufficient statistics, update the prior with them and sample the posterior. It
    knows nothing about comparing variants: ``BayesianTest`` does that.
    """

    @abstractmethod
    def summarize(self, data: Data) -> Statistics:
        """Validate raw data and reduce it to its sufficient statistics."""

    @abstractmethod
    def validate(self, statistics: Statistics) -> None:
        """Raise a ``ValueError`` if the (aggregated) statistics are not valid."""

    @abstractmethod
    def fit_priors(self, historical: Data) -> "ConjugateModel":
        """Set ``prior`` from historical data, at its full weight. Returns ``self``."""

    @abstractmethod
    def weighted(self, weight: float) -> "ConjugateModel":
        """Copy whose prior is worth ``weight`` observations, keeping the prior mean."""

    @abstractmethod
    def n_observations(self, statistics: Statistics) -> float:
        """Number of observations behind the statistics."""

    @abstractmethod
    def observed(self, statistics: Statistics) -> Dict[str, Any]:
        """Observed (data) columns reported by ``evaluate``, in display order."""

    @abstractmethod
    def posterior_samples(
        self, statistics: Statistics, sim_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Posterior draws of the metric compared between variants."""

    @abstractmethod
    def posterior_mean(self, statistics: Statistics) -> float:
        """Posterior mean of the metric, in closed form from the conjugate update."""
