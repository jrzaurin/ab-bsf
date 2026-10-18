from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ab_bfs.frequentist.experiments.base import BaseFrequentistTest, Distribution, TestType
from ab_bfs.utils.utils import alias_params


class DeltaLognormalFrequentistTest(BaseFrequentistTest):
    """
    Frequentist design for zero-inflated, lognormal-ish metrics (revenue or volume
    per session), using the two-part test that mirrors the Bayesian delta-lognormal:

    1. conversion (zero vs non-zero): two-proportion z-test;
    2. magnitude among converters: t-test on log(values).

    The experiment "detects" an effect if either part is significant; with
    ``bonferroni_correction`` each part uses alpha_ / 2.

    Additional keyword arguments
    ----------------------------
    conversion_rate : float (required)
        Baseline share of non-zero observations.
    mean_positive : float (required)
        Baseline mean among converters. ``p1`` must equal conversion_rate * mean_positive.
    std_log : float
        SD of log(values) among converters. Or pass ``std_positive`` (raw-scale SD of
        the positives) and it is converted assuming lognormality.
    conversion_share : float, default 0.5
        For ``sample_size_given_power`` / ``power_given_sample_size``: how a target
        lift on the overall mean is split between the two parts, on the log scale:
        (1 + lift) = (1 + conv_lift)(1 + mag_lift), conv_lift = (1 + lift)^share - 1.
    bonferroni_correction : bool, default True

    Examples
    --------
    >>> test = DeltaLognormalFrequentistTest(
    ...     n1=5000, p1=10.5, conversion_rate=0.3, mean_positive=35, std_log=0.5)
    >>> mde, target = test.minimum_detectable_effect()
    >>> round(mde, 3)
    0.086
    """

    @alias_params(n1=["n_control", "initial_control_sample_size", "n_per_variant"], p1="overall_mean")
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
        ap = self.additional_params
        for param in ("conversion_rate", "mean_positive"):
            if param not in ap:
                raise ValueError(f"{param} must be provided in kwargs")
        self.conversion_rate = float(ap["conversion_rate"])
        self.mean_positive = float(ap["mean_positive"])
        if not 0 < self.conversion_rate < 1:
            raise ValueError("conversion_rate must be strictly between 0 and 1")
        if "std_log" in ap:
            self.std_log = float(ap["std_log"])
        elif "std_positive" in ap:
            self.std_log = float(np.sqrt(np.log(1 + (ap["std_positive"] / self.mean_positive) ** 2)))
        else:
            raise ValueError("Either std_log or std_positive must be provided")
        self.conversion_share = float(ap.get("conversion_share", 0.5))
        if not 0 <= self.conversion_share <= 1:
            raise ValueError("conversion_share must be in [0, 1]")
        self.bonferroni_correction = bool(ap.get("bonferroni_correction", True))
        expected = self.conversion_rate * self.mean_positive
        if not np.isclose(self.p1, expected, rtol=0.01):
            raise ValueError(f"overall_mean ({self.p1}) should equal conversion_rate * mean_positive ({expected})")

    # --------------------------------------------------------------- helpers
    @property
    def _alpha_part(self) -> float:
        return self.alpha_ / 2 if self.bonferroni_correction else self.alpha_

    def _split(self, p2: float) -> Tuple[float, float]:
        """New conversion rate and new mean among converters for an overall target p2."""
        growth = p2 / self.p1
        if growth <= 0:
            raise ValueError("p2 must be positive")
        conv = self.conversion_rate * growth**self.conversion_share
        mag = self.mean_positive * growth ** (1 - self.conversion_share)
        return min(conv, 1 - 1e-9), mag

    # ------------------------------------------------------------------- API
    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(self, control_sample_size: Optional[int] = None) -> Tuple[float, float]:
        """
        Lift on the overall mean detectable with ``power`` if it came *entirely* from
        one part; the larger (more conservative) of the conversion-only and
        magnitude-only MDEs is returned as (relative MDE, target overall mean).
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z = self._z_alpha(n_c, n_t, alpha=self._alpha_part) + norm.ppf(self.power)

        p = self.conversion_rate
        conv_abs = z * np.sqrt(p * (1 - p) * (1 / n_c + 1 / n_t))
        mde_conv = conv_abs / p  # overall mean scales 1:1 with conversion

        k_c, k_t = max(n_c * p, 1.0), max(n_t * p, 1.0)
        mde_mag = float(np.exp(z * self.std_log * np.sqrt(1 / k_c + 1 / k_t)) - 1)

        mde = max(mde_conv, mde_mag)
        return float(mde), float(self.p1 * (1 + mde))

    @alias_params(p2="target_overall_mean")
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """Sizes so that *both* parts reach ``power`` for their share of the lift."""
        z = self._z_alpha(alpha=self._alpha_part) + norm.ppf(self.power)
        conv2, mag2 = self._split(p2)
        r = self.ratio

        n_conv = 0.0
        if self.conversion_share > 0:
            p1c = self.conversion_rate
            var = (p1c * (1 - p1c) + conv2 * (1 - conv2)) / 2
            n_conv = z**2 * var * (1 + 1 / r) / (conv2 - p1c) ** 2

        n_mag = 0.0
        if self.conversion_share < 1:
            effect_log = np.log(mag2 / self.mean_positive)
            n_mag = z**2 * self.std_log**2 * (1 + 1 / r) / effect_log**2 / self.conversion_rate

        n_control = int(np.ceil(max(n_conv, n_mag)))
        return n_control, int(np.ceil(n_control * r))

    @alias_params(p2="target_overall_mean", control_sample_size="total_control_sample_size")
    def power_given_sample_size(self, p2: float, control_sample_size: Optional[int] = None) -> float:
        """P(at least one part significant), treating the two tests as independent."""
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z_a = self._z_alpha(n_c, n_t, alpha=self._alpha_part)
        conv2, mag2 = self._split(p2)

        p1c = self.conversion_rate
        var = (p1c * (1 - p1c) + conv2 * (1 - conv2)) / 2
        pw_conv = self._power_from_z((conv2 - p1c) / np.sqrt(var * (1 / n_c + 1 / n_t)), z_a)

        k_c, k_t = max(n_c * p1c, 1.0), max(n_t * p1c, 1.0)
        z_mag = np.log(mag2 / self.mean_positive) / (self.std_log * np.sqrt(1 / k_c + 1 / k_t))
        pw_mag = self._power_from_z(z_mag, z_a)

        return float(np.clip(pw_conv + pw_mag - pw_conv * pw_mag, 0, 1))
