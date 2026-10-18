"""
ab_bfs: Bayesian and frequentist A/B testing.

    ab_bfs.bayesian.experiments    conjugate tests (binary, normal, delta-lognormal, ...)
    ab_bfs.bayesian.metrics        posterior samplers and decision metrics
    ab_bfs.frequentist.experiments design: power, sample size, MDE, duration
    ab_bfs.frequentist.metrics     evaluation: t-tests, DiD, CUPED, regressions
"""

from . import bayesian, datasets, frequentist, utils

__version__ = "0.2.0"
__all__ = ["bayesian", "frequentist", "datasets", "utils", "__version__"]
