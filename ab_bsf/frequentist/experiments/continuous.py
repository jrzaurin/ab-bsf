from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ab_bsf.frequentist.experiments.base import BaseFrequentistTest, Distribution, TestType
from ab_bsf.utils.utils import alias_params


class ContinuousFrequentistTest(BaseFrequentistTest):
    """
    Frequentist design for continuous metrics (revenue, time spent), Welch-style
    standard error sqrt(std1^2 / n1 + std2^2 / n2).

    Additional keyword arguments
    ----------------------------
    std1 : float (required)
        Standard deviation of the metric in control.
    std2 : float, optional
        Standard deviation in treatment (defaults to std1).
    n_variants : int, optional
        Arms including control (Bonferroni correction).

    Examples
    --------
    >>> test = ContinuousFrequentistTest(n1=1000, mean1=100, std1=20)
    >>> test.test_duration(daily_observations=500, mde=0.1)["duration_days"]
    0.0
    """

    @alias_params(p1="mean1", n1=["n_control", "initial_control_sample_size"])
    def __init__(
        self,
        n1: int,
        p1: float,
        alpha: float = 0.05,
        power: float = 0.8,
        n2: Optional[int] = None,
        ratio: float = 1.0,
        distribution: Distribution = "normal",
        test_type: TestType = "one_tail",
        **kwargs,
    ):
        super().__init__(n1, p1, alpha, power, n2, ratio, distribution, test_type, **kwargs)
        if "std1" not in self.additional_params:
            raise ValueError("std1 must be provided in kwargs")
        self.std1 = float(self.additional_params["std1"])
        self.std2 = float(self.additional_params.get("std2", self.std1))
        if self.std1 <= 0 or self.std2 <= 0:
            raise ValueError("std1 and std2 must be positive")

    def _se(self, n_c: int, n_t: int) -> float:
        return float(np.sqrt(self.std1**2 / n_c + self.std2**2 / n_t))

    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(self, control_sample_size: Optional[int] = None) -> Tuple[float, float]:
        """(relative MDE, mean2_min)."""
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        min_diff = (self._z_alpha(n_c, n_t) + norm.ppf(self.power)) * self._se(n_c, n_t)
        mde = min_diff / self.p1 if self.p1 != 0 else float("inf")
        return float(mde), float(self.p1 + min_diff)

    @alias_params(p2="mean2", control_sample_size="total_control_sample_size")
    def power_given_sample_size(self, p2: float, control_sample_size: Optional[int] = None) -> float:
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        return self._power_from_z((p2 - self.p1) / self._se(n_c, n_t), self._z_alpha(n_c, n_t))

    @alias_params(p2="mean2")
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """n1 = (z_a + z_b)^2 (std1^2 + std2^2 / ratio) / delta^2,  n2 = ratio * n1."""
        delta = abs(p2 - self.p1)
        if delta == 0:
            raise ValueError("p2 equals p1 (zero effect)")
        z = self._z_alpha() + norm.ppf(self.power)
        n_control = int(np.ceil(z**2 * (self.std1**2 + self.std2**2 / self.ratio) / delta**2))
        return n_control, int(np.ceil(n_control * self.ratio))
