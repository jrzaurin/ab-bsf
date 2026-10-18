"""Base class for frequentist experiment *design* (power, sample size, MDE, duration)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.stats import norm, t

from ab_bsf.utils.utils import alias_params

TestType = Literal["one_tail", "two_tail"]
Distribution = Literal["normal", "t"]


class BaseFrequentistTest(ABC):
    """
    Shared machinery for the frequentist design classes.

    Parameters
    ----------
    n1 : int
        Control sample size already available (or planned). MDE and power use it by
        default; ``test_duration`` counts the days needed *beyond* it.
    p1 : float
        Baseline metric: conversion rate (binary) or mean (continuous).
    alpha, power : float
        Significance level and target power.
    n2 : int, optional
        Treatment sample size. Defaults to ``n1 * ratio``.
    ratio : float
        Treatment / control size.
    distribution : "normal" | "t"
        Critical values from the normal or t distribution (power always uses the
        normal approximation for the non-centrality).
    test_type : "one_tail" | "two_tail"
    **kwargs
        Class-specific parameters, stored in ``self.additional_params``.
        ``n_variants`` (total arms, including control) turns on Bonferroni
        correction: ``alpha_ = alpha / (n_variants - 1)``; equal groups only.
    """

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
        **kwargs: Any,
    ) -> None:
        if n1 <= 0:
            raise ValueError("n1 must be positive")
        if not 0 < alpha < 1 or not 0 < power < 1:
            raise ValueError("alpha and power must be between 0 and 1")
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        if test_type not in ("one_tail", "two_tail"):
            raise ValueError("test_type must be 'one_tail' or 'two_tail'")
        if distribution not in ("normal", "t"):
            raise ValueError("distribution must be 'normal' or 't'")

        self.n1 = int(n1)
        self.p1 = p1
        self.alpha = alpha
        self.power = power
        self.ratio = ratio
        self.n2 = int(n2) if n2 is not None else int(round(n1 * ratio))
        self.distribution = distribution
        self.test_type = test_type
        self.additional_params: Dict[str, Any] = dict(kwargs)

        if "n_variants" in kwargs:
            self.n_variants = int(kwargs["n_variants"])
            if self.n_variants < 2:
                raise ValueError("n_variants must be at least 2")
            if self.n2 != self.n1 or ratio != 1.0:
                raise ValueError("n_variants is not supported for non-equal groups")
            self.alpha_ = alpha / (self.n_variants - 1)
        else:
            self.alpha_ = alpha

    # ----------------------------------------------------------------- helpers
    def _overwrite_sample_size(self, sample_size: Optional[int] = None) -> Tuple[int, int]:
        """(control, treatment) sizes: the stored ones, or ``sample_size`` and ``sample_size * ratio``."""
        if sample_size is None:
            return self.n1, self.n2
        return int(sample_size), max(1, int(np.ceil(sample_size * self.ratio - 1e-9)))

    def _z_alpha(
        self, n_control: Optional[int] = None, n_treatment: Optional[int] = None, alpha: Optional[float] = None
    ) -> float:
        a = self.alpha_ if alpha is None else alpha
        q = 1 - a / 2 if self.test_type == "two_tail" else 1 - a
        if self.distribution == "t" and n_control is not None and n_treatment is not None:
            return float(t.ppf(q, n_control + n_treatment - 2))
        return float(norm.ppf(q))

    def _power_from_z(self, z_stat: float, z_alpha: float) -> float:
        """Power for a test statistic with mean ``z_stat`` (absolute value) under H1."""
        z_stat = abs(z_stat)
        power = 1 - norm.cdf(z_alpha - z_stat)
        if self.test_type == "two_tail":
            power += norm.cdf(-z_alpha - z_stat)
        return float(power)

    # --------------------------------------------------------------- abstract
    @abstractmethod
    def minimum_detectable_effect(self, control_sample_size: Optional[int] = None) -> Tuple[float, float]:
        """(relative MDE, target metric) detectable with ``power`` at the given size."""

    @abstractmethod
    def power_given_sample_size(self, p2: float, control_sample_size: Optional[int] = None) -> float:
        """Power to detect a move from ``p1`` to ``p2``."""

    @abstractmethod
    def sample_size_given_power(self, p2: float) -> Tuple[int, int]:
        """(control, treatment) sizes needed to detect ``p1`` -> ``p2`` with ``power``."""

    # ----------------------------------------------------------- multivariant
    def _require_variants(self) -> None:
        if not hasattr(self, "n_variants"):
            raise ValueError("Pass n_variants=... when creating the test to use multivariant methods")

    def sample_size_given_power_multivariant(self, p2: float) -> Tuple[int, List[int]]:
        """Control size and the list of treatment sizes (Bonferroni-corrected alpha)."""
        self._require_variants()
        n_control, n_treatment = self.sample_size_given_power(p2)
        return n_control, [n_treatment] * (self.n_variants - 1)

    def power_given_sample_size_multivariant(
        self, p2: float, control_sample_size: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        (per-comparison power, P(at least one arm significant)) when every treatment
        arm has the effect, assuming independent comparisons.
        """
        self._require_variants()
        per = self.power_given_sample_size(p2, control_sample_size)
        return per, 1 - (1 - per) ** (self.n_variants - 1)

    # --------------------------------------------------------------- duration
    def _target(self, mde: Optional[float]) -> Tuple[float, float]:
        if mde is None:
            mde = self.minimum_detectable_effect()[0]
        return mde, self.p1 * (1 + mde)

    def test_duration(
        self,
        daily_observations: int,
        mde: Optional[float] = None,
        already_collected: bool = True,
    ) -> Dict[str, Any]:
        """
        Days needed to detect a relative lift ``mde``.

        Parameters
        ----------
        daily_observations : new observations per day for the *control* group
            (treatment groups get ``daily_observations * ratio``).
        mde : relative lift to detect (e.g. 0.05 = +5%). Defaults to the current MDE.
        already_collected : if True, the ``n1`` control observations given at
            construction count towards the requirement; days never go negative.

        Returns
        -------
        dict with ``duration_days``, required sizes, ``target_mean``,
        ``mde_relative`` and ``mde_absolute``.
        """
        if daily_observations <= 0:
            raise ValueError("daily_observations must be positive")
        mde, target = self._target(mde)
        if hasattr(self, "n_variants"):
            n_control, n_treatments = self.sample_size_given_power_multivariant(target)
            n_treatment = n_treatments[0]
        else:
            n_control, n_treatment = self.sample_size_given_power(target)
        have = self.n1 if already_collected else 0
        days = max(0.0, float(np.ceil((n_control - have) / daily_observations)))
        return {
            "duration_days": days,
            "required_control_sample_size": n_control,
            "required_treatment_sample_size": n_treatment,
            "target_mean": target,
            "mde_relative": mde,
            "mde_absolute": target - self.p1,
        }

    def power_over_time(self, daily_observations: int, p2: float, max_days: int = 60) -> Tuple[List[int], List[float]]:
        """Power after each day when control grows by ``daily_observations`` per day (from zero)."""
        days = list(range(1, max_days + 1))
        return days, [self.power_given_sample_size(p2, daily_observations * d) for d in days]

    def days_for_target_power(
        self, daily_observations: int, p2: float, target_power: Optional[float] = None, max_days: int = 365
    ) -> Optional[int]:
        """First day on which power reaches ``target_power`` (default: ``self.power``), or None."""
        target_power = self.power if target_power is None else target_power
        days, powers = self.power_over_time(daily_observations, p2, max_days)
        return next((d for d, pw in zip(days, powers) if pw >= target_power), None)

    def mde_given_duration(self, daily_observations: int, days: int) -> Dict[str, Any]:
        """
        What can be detected in a fixed number of days: the MDE at ``self.power``
        with ``daily_observations * days`` control observations.
        """
        n = int(daily_observations * days)
        mde, target = self.minimum_detectable_effect(control_sample_size=n)
        return {
            "duration_days": days,
            "control_sample_size": n,
            "mde_relative": mde,
            "target_mean": target,
            "power": self.power,
        }


__all__ = ["BaseFrequentistTest", "alias_params"]
