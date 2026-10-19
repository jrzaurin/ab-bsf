from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ab_bsf.frequentist.experiments.base import (
    BaseFrequentistTest,
    Distribution,
    TestType,
)
from ab_bsf.utils.utils import alias_params


class BinaryFrequentistTest(BaseFrequentistTest):
    """
    Frequentist design for conversion-rate experiments.

    The design rests on the normal approximation of the difference in
    conversion rates. With ``n_c`` control and ``n_t = ratio * n_c`` treatment
    observations, and the same per-observation variance ``var`` in both arms:

        SE = sqrt(var * (1/n_c + 1/n_t))
        Z  = (p2_hat - p1_hat) / SE

    Under H0 (no effect) ``Z ~ N(0, 1)``, and H0 is rejected when ``Z`` exceeds
    the critical value ``z_a`` (see ``_z_alpha``). If the true rates are ``p1``
    and ``p2``, ``Z ~ N(z_stat, 1)`` with ``z_stat = (p2 - p1) / SE``, so

        power = P(Z > z_a) = 1 - Phi(z_a - z_stat)

    (two-tailed tests add the opposite tail, ``Phi(-z_a - z_stat)``). The three
    methods solve this one relation for different unknowns:

    - ``power_given_sample_size``: compute ``z_stat`` and evaluate the power.
    - ``sample_size_given_power``: require ``z_stat = z_a + z_b``, with
      ``z_b = Phi^-1(power)``, and solve for ``n_c``.
    - ``minimum_detectable_effect``: require ``z_stat = z_a + z_b``, and solve
      for ``p2``.

    Parameters
    ----------
    p1_observed_var : float
        Empirical per-observation variance of the metric, required. It is used
        for both arms.
    n_variants : int, optional
        Number of arms including control (Bonferroni correction).

    Examples
    --------
    >>> test = BinaryFrequentistTest(n1=1000, p1=0.05, p1_observed_var=0.0475)
    >>> test.sample_size_given_power(p2=0.10)
    (235, 235)
    """

    @alias_params(
        p1=["conversion_rate", "baseline_conversion"],
        n1=["n_control", "initial_control_sample_size"],
    )
    def __init__(
        self,
        n1: int,
        p1: float,
        p1_observed_var: float,
        alpha: float = 0.05,
        power: float = 0.8,
        n2: Optional[int] = None,
        ratio: float = 1.0,
        distribution: Distribution = "normal",
        test_type: TestType = "one_tail",
        **kwargs,
    ):
        if not 0 < p1 < 1:
            raise ValueError("p1 must be a conversion rate strictly between 0 and 1")
        if p1_observed_var <= 0:
            raise ValueError("p1_observed_var must be positive")
        super().__init__(
            n1, p1, alpha, power, n2, ratio, distribution, test_type, **kwargs
        )
        self.p1_observed_var = p1_observed_var

    @alias_params(p2=["target_conversion", "conversion_rate_2"])
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """
        Sample sizes to detect ``p1 -> p2``.

        Substituting ``n_t = ratio * n_c`` in ``(p2 - p1) / SE = z_a + z_b`` gives
        ``1/n_c + 1/n_t = (1 + 1/ratio) / n_c``, hence

            n_c = (z_a + z_b)^2 * var * (1 + 1/ratio) / (p2 - p1)^2

        ``n_c`` is unknown here, so ``z_a`` always comes from the normal
        distribution, and the opposite tail of a two-tailed test is ignored.
        """
        if not 0 < p2 < 1:
            raise ValueError("p2 must be strictly between 0 and 1")
        if abs(p2 - self.p1) < 1e-12:
            raise ValueError("p1 and p2 are too similar (zero effect)")
        z = self._z_alpha() + norm.ppf(self.power)
        n_control = (
            z**2 * self.p1_observed_var * (1 + 1 / self.ratio) / (p2 - self.p1) ** 2
        )
        n_control = int(np.ceil(n_control))
        return n_control, int(np.ceil(n_control * self.ratio))

    @alias_params(
        p2=["target_conversion", "conversion_rate_2"],
        control_sample_size="total_control_sample_size",
    )
    def power_given_sample_size(
        self, p2: float, control_sample_size: Optional[int] = None
    ) -> float:
        """
        Power to detect ``p1 -> p2`` with the stored or given control size.

        ``z_stat`` is the expected value of ``Z`` when the true rates are ``p1``
        and ``p2``: the effect ``p2 - p1`` measured in standard errors.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z_a = self._z_alpha(n_c, n_t)
        z_stat = (p2 - self.p1) / np.sqrt(self.p1_observed_var * (1 / n_c + 1 / n_t))
        return self._power_from_z(z_stat, z_a)

    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(
        self, control_sample_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Smallest upward lift detectable with ``power``: (relative MDE, p2_min).

        Solving ``(p2 - p1) / SE = z_a + z_b`` for ``p2`` gives

            p2 = p1 + (z_a + z_b) * SE

        capped at 1. The relative MDE is ``(p2 - p1) / p1``.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z = self._z_alpha(n_c, n_t) + norm.ppf(self.power)
        p2 = min(self.p1 + z * np.sqrt(self.p1_observed_var * (1 / n_c + 1 / n_t)), 1.0)
        return (p2 - self.p1) / self.p1, float(p2)
