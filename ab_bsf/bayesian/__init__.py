"""Bayesian A/B testing with conjugate models, inspired by Matus Baniar's bayesian-testing."""

from . import experiments, metrics, models
from .experiments import *  # noqa: F401,F403
from .experiments import __all__

__all__ = list(__all__) + ["experiments", "metrics", "models"]
