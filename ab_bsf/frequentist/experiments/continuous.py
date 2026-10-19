from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ab_bsf.frequentist.experiments.base import (
    BaseFrequentistTest,
    Distribution,
    TestType,
)
from ab_bsf.utils.utils import alias_params


class ContinuousFrequentistTest(BaseFrequentistTest):
    """
    Frequentist design for continuous metrics (revenue, time spent), comparing
    the means of control and treatment.

    The design rests on the normal approximation of the difference in means.
    With ``n_c`` control and ``n_t = ratio * n_c`` treatment observations, and
    standard deviations ``std1`` and ``std2`` (Welch-style, so they may differ):

        SE = sqrt(std1^2 / n_c + std2^2 / n_t)
        Z  = (mean2_hat - mean1_hat) / SE

    Under H0 (no effect) ``Z ~ N(0, 1)``, and H0 is rejected when ``Z`` exceeds
    the critical value ``z_a`` (see ``_z_alpha``). If the true means are ``p1``
    and ``p2``, ``Z ~ N(z_stat, 1)`` with ``z_stat = (p2 - p1) / SE``, so

        power = P(Z > z_a) = 1 - Phi(z_a - z_stat)

    (two-tailed tests add the opposite tail, ``Phi(-z_a - z_stat)``). The three
    methods solve this one relation for different unknowns:

    - ``power_given_sample_size``: compute ``z_stat`` and evaluate the power.
    - ``sample_size_given_power``: require ``z_stat = z_a + z_b``, with
      ``z_b = Phi^-1(power)``, and solve for ``n_c``.
    - ``minimum_detectable_effect``: require ``z_stat = z_a + z_b``, and solve
      for the difference in means.

    Here ``p1`` is the control mean. It is also accepted as ``mean1``.

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
    >>> # A smaller lift needs 1237 control observations, 237 beyond the 1000 already
    >>> # collected, i.e. 3 days at 100 per day:
    >>> test.test_duration(daily_observations=100, mde=0.02)["duration_days"]
    3.0
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
        super().__init__(
            n1, p1, alpha, power, n2, ratio, distribution, test_type, **kwargs
        )
        if "std1" not in self.additional_params:
            raise ValueError("std1 must be provided in kwargs")
        self.std1 = float(self.additional_params["std1"])
        self.std2 = float(self.additional_params.get("std2", self.std1))
        if self.std1 <= 0 or self.std2 <= 0:
            raise ValueError("std1 and std2 must be positive")

    def _se(self, n_c: int, n_t: int) -> float:
        """Standard error of the difference in means."""
        return float(np.sqrt(self.std1**2 / n_c + self.std2**2 / n_t))

    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(
        self, control_sample_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Smallest upward lift detectable with ``power``: (relative MDE, mean2_min).

        Solving ``(mean2 - p1) / SE = z_a + z_b`` for the difference gives

            mean2 - p1 = (z_a + z_b) * SE

        The relative MDE is that difference divided by ``p1``.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        min_diff = (self._z_alpha(n_c, n_t) + norm.ppf(self.power)) * self._se(n_c, n_t)
        mde = min_diff / self.p1 if self.p1 != 0 else float("inf")
        return float(mde), float(self.p1 + min_diff)

    @alias_params(p2="mean2", control_sample_size="total_control_sample_size")
    def power_given_sample_size(
        self, p2: float, control_sample_size: Optional[int] = None
    ) -> float:
        """
        Power to detect ``p1 -> p2`` with the stored or given control size.

        ``z_stat = (p2 - p1) / SE`` is the effect measured in standard errors.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        return self._power_from_z(
            (p2 - self.p1) / self._se(n_c, n_t), self._z_alpha(n_c, n_t)
        )

    @alias_params(p2="mean2")
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """
        Sample sizes to detect ``p1 -> p2``.

        Substituting ``n_t = ratio * n_c`` in ``|p2 - p1| / SE = z_a + z_b`` gives
        ``SE^2 = (std1^2 + std2^2 / ratio) / n_c``, hence

            n_c = (z_a + z_b)^2 * (std1^2 + std2^2 / ratio) / (p2 - p1)^2
            n_t = ratio * n_c

        ``n_c`` is unknown here, so ``z_a`` always comes from the normal
        distribution, and the opposite tail of a two-tailed test is ignored.
        """
        delta = abs(p2 - self.p1)
        if delta == 0:
            raise ValueError("p2 equals p1 (zero effect)")
        z = self._z_alpha() + norm.ppf(self.power)
        n_control = int(
            np.ceil(z**2 * (self.std1**2 + self.std2**2 / self.ratio) / delta**2)
        )
        return n_control, int(np.ceil(n_control * self.ratio))
