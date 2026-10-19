from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ab_bsf.frequentist.experiments.base import (
    BaseFrequentistTest,
    Distribution,
    TestType,
)
from ab_bsf.utils.utils import alias_params


class DeltaLognormalFrequentistTest(BaseFrequentistTest):
    """
    Frequentist design for zero-inflated, lognormal-ish metrics (revenue or volume
    per session), using a two-part test that mirrors the Bayesian delta-lognormal.

    Model
    -----
    Each observation is 0 with probability ``1 - p`` and, otherwise, lognormal
    with log-scale mean ``mu`` and SD ``sigma``. The overall mean (``p1``) is
    therefore ``p * m``, with ``m = exp(mu + sigma^2 / 2)`` the mean among
    converters (``mean_positive``). A lift on the overall mean comes from two
    places: more converters (``p``) or larger values among converters (``m``).

    Two tests
    ---------
    1. Conversion (zero vs non-zero): two-proportion z-test,

           SE_conv = sqrt(p (1 - p) * (1/n_c + 1/n_t))
           z_stat  = (p2 - p) / SE_conv

       using the baseline variance ``p (1 - p)`` for both arms.

    2. Magnitude (among converters): t-test on ``log(values)``. If ``sigma`` is
       unchanged, a change in ``m`` from ``m1`` to ``m2`` shifts the mean of the
       logs by ``log(m2 / m1)``, so

           SE_mag = sigma * sqrt(1/k_c + 1/k_t)
           z_stat = log(m2 / m1) / SE_mag

       where ``k = n * p`` is the expected number of converters in each arm
       (the baseline ``p`` is used for both arms: conservative when the
       treatment raises conversion).

    Both tests use the normal-approximation relation of the other design classes,
    ``power = 1 - Phi(z_a - z_stat)`` (plus the opposite tail if two-tailed), with
    ``z_a`` at the per-part significance level ``alpha_ / 2`` (Bonferroni,
    because the experiment "detects" an effect if *either* part is significant;
    set ``bonferroni_correction=False`` to use ``alpha_`` for each).

    The three methods answer slightly different questions, so they are not exact
    inverses of each other:

    - ``minimum_detectable_effect``: the lift detectable if it came *entirely*
      from either one part (the larger of the two is returned). A lift split
      between both parts is not guaranteed to reach ``power``.
    - ``sample_size_given_power``: sizes so that *both* parts reach ``power`` for
      their share of the lift.
    - ``power_given_sample_size``: probability that *at least one* part is
      significant. At the sizes from ``sample_size_given_power`` this is
      therefore above ``power``.

    Additional keyword arguments
    ----------------------------
    conversion_rate : float (required)
        Baseline share of non-zero observations, ``p``.
    mean_positive : float (required)
        Baseline mean among converters, ``m``. ``p1`` (also accepted as
        ``overall_mean``) must equal conversion_rate * mean_positive.
    std_log : float
        SD of log(values) among converters, ``sigma``. Or pass ``std_positive``
        (raw-scale SD of the positives): it is converted assuming lognormality,
        ``sigma^2 = log(1 + (std_positive / mean_positive)^2)``.
    conversion_share : float, default 0.5
        How a target lift on the overall mean is split between the two parts,
        on the log scale: with ``growth = p2 / p1``, conversion grows by
        ``growth^share`` and the converters' mean by ``growth^(1 - share)``, so
        the two growths multiply back to ``growth``. ``1`` puts the whole lift
        on conversion, ``0`` on magnitude.
    bonferroni_correction : bool, default True
        Use ``alpha_ / 2`` for each part instead of ``alpha_``.

    Examples
    --------
    >>> test = DeltaLognormalFrequentistTest(
    ...     n1=5000, p1=10.5, conversion_rate=0.3, mean_positive=35, std_log=0.5)
    >>> mde, target = test.minimum_detectable_effect()
    >>> round(mde, 3)
    0.086
    """

    @alias_params(
        n1=["n_control", "initial_control_sample_size", "n_per_variant"],
        p1="overall_mean",
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
        super().__init__(
            n1, p1, alpha, power, n2, ratio, distribution, test_type, **kwargs
        )
        ap = self.additional_params
        for param in ("conversion_rate", "mean_positive"):
            if param not in ap:
                raise ValueError(f"{param} must be provided in kwargs")
        self.conversion_rate = float(ap["conversion_rate"])
        self.mean_positive = float(ap["mean_positive"])
        if not 0 < self.conversion_rate < 1:
            raise ValueError("conversion_rate must be strictly between 0 and 1")
        self.conversion_var = self.conversion_rate * (1 - self.conversion_rate)
        if "std_log" in ap:
            self.std_log = float(ap["std_log"])
        elif "std_positive" in ap:
            self.std_log = float(
                np.sqrt(np.log(1 + (ap["std_positive"] / self.mean_positive) ** 2))
            )
        else:
            raise ValueError("Either std_log or std_positive must be provided")
        self.conversion_share = float(ap.get("conversion_share", 0.5))
        if not 0 <= self.conversion_share <= 1:
            raise ValueError("conversion_share must be in [0, 1]")
        self.bonferroni_correction = bool(ap.get("bonferroni_correction", True))
        expected = self.conversion_rate * self.mean_positive
        if not np.isclose(self.p1, expected, rtol=0.01):
            raise ValueError(
                f"overall_mean ({self.p1}) should equal conversion_rate * mean_positive ({expected})"
            )

    @alias_params(control_sample_size="total_control_sample_size")
    def minimum_detectable_effect(
        self, control_sample_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Smallest relative lift on the overall mean detectable with ``power``,
        returned as (relative MDE, target overall mean).

        Each part is solved on its own, as in the other design classes,
        ``|effect| / SE = z_a + z_b``, assuming the whole lift comes from that part:

        - conversion: ``p2 - p = (z_a + z_b) * SE_conv``. The overall mean is
          proportional to ``p``, so the relative lift is ``(p2 - p) / p``.
        - magnitude: ``log(m2 / m) = (z_a + z_b) * SE_mag``, so the relative lift
          (``m2 / m - 1``) is ``exp((z_a + z_b) * SE_mag) - 1``.

        The larger of the two is returned: a lift of that size reaches ``power``
        if it comes entirely from either part. It is *not* guaranteed when the
        lift is split between both parts, since each then gets only a fraction
        of the effect (e.g. with ``conversion_share=0.75`` the power at the
        returned MDE can fall below ``power``).
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z = self._z_alpha(n_c, n_t, alpha=self._alpha_part) + norm.ppf(self.power)

        p = self.conversion_rate
        conv_abs = z * np.sqrt(self.conversion_var * (1 / n_c + 1 / n_t))
        mde_conv = conv_abs / p

        k_c, k_t = max(n_c * p, 1.0), max(n_t * p, 1.0)
        mde_mag = float(np.exp(z * self.std_log * np.sqrt(1 / k_c + 1 / k_t)) - 1)

        mde = max(mde_conv, mde_mag)
        return float(mde), float(self.p1 * (1 + mde))

    @alias_params(p2="target_overall_mean")
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """
        Sizes so that *both* parts reach ``power`` for their share of the lift
        (see ``_split``). Each part is sized as in the other design classes,
        solving ``|effect| / SE = z_a + z_b`` for the control size, with
        ``n_t = ratio * n_c``, and the larger of the two is returned:

        - conversion: ``n_c = z^2 * p (1 - p) * (1 + 1/ratio) / (conv2 - p)^2``.
        - magnitude: the required number of control converters is
          ``k_c = z^2 * sigma^2 * (1 + 1/ratio) / log(m2 / m)^2``, and since
          ``k_c = n_c * p``, ``n_c = k_c / p``.

        A part that receives none of the lift (``conversion_share`` of 0 or 1)
        needs no sample.
        """
        if p2 == self.p1:
            raise ValueError("p2 equals p1 (zero effect)")
        z = self._z_alpha(alpha=self._alpha_part) + norm.ppf(self.power)
        conv2, mag2 = self._split(p2)
        r = self.ratio

        n_conv = 0.0
        if self.conversion_share > 0:
            n_conv = (
                z**2
                * self.conversion_var
                * (1 + 1 / r)
                / (conv2 - self.conversion_rate) ** 2
            )

        n_mag = 0.0
        if self.conversion_share < 1:
            effect_log = np.log(mag2 / self.mean_positive)
            n_mag = (
                z**2
                * self.std_log**2
                * (1 + 1 / r)
                / effect_log**2
                / self.conversion_rate
            )

        n_control = int(np.ceil(max(n_conv, n_mag)))
        return n_control, int(np.ceil(n_control * r))

    @alias_params(
        p2="target_overall_mean", control_sample_size="total_control_sample_size"
    )
    def power_given_sample_size(
        self, p2: float, control_sample_size: Optional[int] = None
    ) -> float:
        """
        Probability that at least one part is significant, with the lift split
        between the parts as in ``_split``.

        Each part has power ``1 - Phi(z_a - z_stat)`` (plus the opposite tail if
        two-tailed), with ``z_stat`` as in the class docstring. Treating the two
        tests as independent, ``P(at least one) = 1 - (1 - pw_conv)(1 - pw_mag)``.
        """
        n_c, n_t = self._overwrite_sample_size(control_sample_size)
        z_a = self._z_alpha(n_c, n_t, alpha=self._alpha_part)
        conv2, mag2 = self._split(p2)

        p1c = self.conversion_rate
        pw_conv = self._power_from_z(
            (conv2 - p1c) / np.sqrt(self.conversion_var * (1 / n_c + 1 / n_t)), z_a
        )

        k_c, k_t = max(n_c * p1c, 1.0), max(n_t * p1c, 1.0)
        z_mag = np.log(mag2 / self.mean_positive) / (
            self.std_log * np.sqrt(1 / k_c + 1 / k_t)
        )
        pw_mag = self._power_from_z(z_mag, z_a)

        return float(np.clip(pw_conv + pw_mag - pw_conv * pw_mag, 0, 1))

    @property
    def _alpha_part(self) -> float:
        """Significance level of each of the two tests."""
        return self.alpha_ / 2 if self.bonferroni_correction else self.alpha_

    def _split(self, p2: float) -> Tuple[float, float]:
        """
        New conversion rate and new mean among converters for an overall target p2.

        The overall growth ``p2 / p1`` is split multiplicatively:
        ``conv = p * growth^share`` and ``mag = m * growth^(1 - share)``.
        """
        growth = p2 / self.p1
        if growth <= 0:
            raise ValueError("p2 must be positive")
        conv = self.conversion_rate * growth**self.conversion_share
        mag = self.mean_positive * growth ** (1 - self.conversion_share)
        return min(conv, 1 - 1e-9), mag
