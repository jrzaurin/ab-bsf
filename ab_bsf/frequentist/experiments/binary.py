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

    Variance options (additional keyword arguments)
    -----------------------------------------------
    p1_observed_var : float, optional
        Empirical per-observation variance; overrides everything else.
    use_pooled_variance : bool, default True
        Pooled binomial variance (p1(1-p1) + p2(1-p2)) / 2. If False, the arcsine
        (Cohen's h) transformation is used.
    n_variants : int, optional
        Number of arms including control (Bonferroni correction).

    Examples
    --------
    >>> test = BinaryFrequentistTest(n1=1000, p1=0.05)
    >>> test.sample_size_given_power(p2=0.10)
    (341, 341)
    """

    @alias_params(
        p1=["conversion_rate", "baseline_conversion"],
        n1=["n_control", "initial_control_sample_size"],
    )
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
        if not 0 < p1 < 1:
            raise ValueError("p1 must be a conversion rate strictly between 0 and 1")
        super().__init__(
            n1, p1, alpha, power, n2, ratio, distribution, test_type, **kwargs
        )
        self.p1_observed_var = self.additional_params.get("p1_observed_var")
        self.use_pooled_variance = self.additional_params.get(
            "use_pooled_variance", True
        )

    # --------------------------------------------------------------- helpers
    def _var(self, p2: float) -> Optional[float]:
        """Per-observation variance on the raw scale, or None for the arcsine method."""
        if self.p1_observed_var is not None:
            return self.p1_observed_var
        if self.use_pooled_variance:
            return (self.p1 * (1 - self.p1) + p2 * (1 - p2)) / 2
        return None

    @staticmethod
    def _cohens_h(p1: float, p2: float) -> float:
        return 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

    # ------------------------------------------------------------------- API
    @alias_params(p2=["target_conversion", "conversion_rate_2"])
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """
        Sample sizes to detect ``p1 -> p2``.

        n1 = (z_a + z_b)^2 * var * (1 + 1/ratio) / (p2 - p1)^2     (pooled / observed)
        n1 = (z_a + z_b)^2 * (1 + 1/ratio) / h^2                   (arcsine, Cohen's h)
        """
        if not 0 < p2 < 1:
            raise ValueError("p2 must be strictly between 0 and 1")
        if abs(p2 - self.p1) < 1e-12:
            raise ValueError("p1 and p2 are too similar (zero effect)")
        z = self._z_alpha() + norm.ppf(self.power)
        var = self._var(p2)
        if var is not None:
            n_control = z**2 * var * (1 + 1 / self.ratio) / (p2 - self.p1) ** 2
        else:
            n_control = z**2 * (1 + 1 / self.ratio) / self._cohens_h(self.p1, p2) ** 2
        n_control = int(np.ceil(n_control))
        return n_control, int(np.ceil(n_control * self.ratio))

    @alias_params(
        p2=["target_conversion", "conversion_rate_2"],
        control_sample_size="total_control_sample_size",
    )
    def power_given_sample_size(
        self, p2: float, control_sample_size: Optional[int] = None
    ) -> float:
        """Power to detect ``p1 -> p2`` with the stored or given control size."""
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z_a = self._z_alpha(n_c, n_t)
        var = self._var(p2)
        if var is not None:
            z_stat = (p2 - self.p1) / np.sqrt(var * (1 / n_c + 1 / n_t))
        else:
            z_stat = self._cohens_h(self.p1, p2) / np.sqrt(1 / n_c + 1 / n_t)
        return self._power_from_z(z_stat, z_a)

    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(
        self, control_sample_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Smallest upward lift detectable with ``power``: returns (relative MDE, p2_min).
        Pooled variance depends on p2, so it is solved by fixed-point iteration.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z = self._z_alpha(n_c, n_t) + norm.ppf(self.power)
        inv_n = 1 / n_c + 1 / n_t
        p1 = self.p1
        if self.p1_observed_var is not None:
            p2 = p1 + z * np.sqrt(self.p1_observed_var * inv_n)
        elif self.use_pooled_variance:
            p2 = p1 + z * np.sqrt(p1 * (1 - p1) * inv_n)
            for _ in range(200):
                p2c = min(p2, 1 - 1e-12)
                new = p1 + z * np.sqrt((p1 * (1 - p1) + p2c * (1 - p2c)) / 2 * inv_n)
                if abs(new - p2) < 1e-13:
                    break
                p2 = new
        else:
            a = np.arcsin(np.sqrt(p1)) + z * np.sqrt(inv_n) / 2
            p2 = 1.0 if a >= np.pi / 2 else np.sin(a) ** 2
        p2 = float(min(p2, 1.0))
        return (p2 - p1) / p1, p2
