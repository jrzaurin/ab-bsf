"""
ab_bsf: Bayesian and frequentist A/B testing.

    ab_bsf.bayesian.experiments    conjugate tests (binary, normal, delta-lognormal, ...)
    ab_bsf.bayesian.metrics        posterior samplers and decision metrics
    ab_bsf.frequentist.experiments design: power, sample size, MDE, duration
    ab_bsf.frequentist.metrics     evaluation: t-tests, DiD, CUPED, regressions
"""

from . import bayesian, datasets, frequentist, utils

__version__ = "0.2.0"
__all__ = ["bayesian", "frequentist", "datasets", "utils", "__version__"]
