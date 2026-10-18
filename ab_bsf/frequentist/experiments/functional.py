"""
Function-style power analysis, MDE and test-duration planning for binary
(conversion) A/B tests -- the July-2025 API that preceded the design classes.

Reconstructed from the July-August 2025 library functions:

    calculate_sample_size_given_power   (earlier name: calculate_sample_size_for_power)
    calculate_power_given_sample_size
    minimum_detectable_effect
    estimate_test_duration
    estimate_test_duration_from_mde
    power_duration_tradeoff_analysis
    calculate_power_over_time
    find_days_for_target_power

Fixes relative to the July 2025 versions
-----------------------------------------
1. Arcsine branch: the "1/4 factor" was wrong and is removed. Cohen's
   h = 2*(asin(sqrt(p2)) - asin(sqrt(p1))) already includes the factor 2, so
   Var(2*asin(sqrt(p_hat))) = 1/n and
       n1 = (z_a + z_b)^2 * (1 + 1/ratio) / h^2
   The 1/4 version under-sized tests 4x (~30% real power instead of 80%).
2. power_duration_tradeoff_analysis: MDE uses a fixed power (default 0.8).
   Using the achievable power just returned the target effect (circular).
3. power_duration_tradeoff_analysis: "target_detectable" comparison was
   inverted (target >= MDE means detectable).
4. Two-tailed power includes the (tiny) opposite-tail rejection probability.
5. Duration functions count users, not conversions.

Variance options (same in every function)
-----------------------------------------
- p1_observed_var given -> empirical per-observation variance, both groups.
- use_pooled_variance=True (default) -> (p1(1-p1) + p2(1-p2)) / 2.
- use_pooled_variance=False -> arcsine (Cohen's h) transformation.
"""

from typing import Literal, Optional

import numpy as np
from scipy.stats import norm, t

TestType = Literal["one_tail", "two_tail"]
Distribution = Literal["normal", "t"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _z_alpha(
    alpha: float,
    test_type: TestType,
    distribution: Distribution = "normal",
    dof: Optional[int] = None,
) -> float:
    q = 1 - alpha / 2 if test_type == "two_tail" else 1 - alpha
    if distribution == "t":
        if dof is None:
            raise ValueError("dof required for the t distribution")
        return float(t.ppf(q, dof))
    return float(norm.ppf(q))


def _cohens_h(p1: float, p2: float) -> float:
    return 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))


def _validate_rates(*ps: float) -> None:
    for p in ps:
        if not 0 < p < 1:
            raise ValueError("Conversion rates must be strictly between 0 and 1")


def _validate_prob(name: str, x: float) -> None:
    if not 0 < x < 1:
        raise ValueError(f"{name} must be between 0 and 1")


# ---------------------------------------------------------------------------
# Sample size given power
# ---------------------------------------------------------------------------
def calculate_sample_size_given_power(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    test_type: TestType = "one_tail",
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
) -> tuple[int, int]:
    """
    Required sample size per group to detect p1 -> p2 with the given power.

    Parameters
    ----------
    p1, p2: float
        Baseline and expected conversion rates.
    alpha: float
        Significance level.
    power: float
        Desired power (1 - beta).
    ratio: float
        Treatment / control size (n2 / n1).
    test_type: "one_tail" | "two_tail"
    p1_observed_var: float, optional
        Empirical per-observation variance; overrides the other options.
    use_pooled_variance: bool
        Pooled binomial variance if True, arcsine transformation if False.

    Returns
    -------
    (n_control, n_treatment)

    Examples
    --------
    >>> calculate_sample_size_given_power(0.05, 0.10)               # one tail
    (341, 341)
    >>> calculate_sample_size_given_power(0.10, 0.12, test_type="two_tail",
    ...                                   use_pooled_variance=False)
    (3835, 3835)
    """
    _validate_rates(p1, p2)
    _validate_prob("alpha", alpha)
    _validate_prob("power", power)
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if abs(p2 - p1) < 1e-12:
        raise ValueError("p1 and p2 are too similar (zero effect)")

    z = _z_alpha(alpha, test_type) + norm.ppf(power)

    if p1_observed_var is not None or use_pooled_variance:
        var = (
            p1_observed_var
            if p1_observed_var is not None
            else (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        )
        n1 = z**2 * var * (1 + 1 / ratio) / (p2 - p1) ** 2
    else:
        n1 = z**2 * (1 + 1 / ratio) / _cohens_h(p1, p2) ** 2

    n1 = int(np.ceil(n1))
    return n1, int(np.ceil(n1 * ratio))


# ---------------------------------------------------------------------------
# Power given sample size
# ---------------------------------------------------------------------------
def calculate_power_given_sample_size(
    n1: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    n2: Optional[int] = None,
    test_type: TestType = "one_tail",
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
    distribution: Distribution = "normal",
) -> float:
    """
    Power to detect p1 -> p2 with n1 (control) and n2 (treatment) observations.
    Inverse of calculate_sample_size_given_power.

    Examples
    --------
    >>> round(calculate_power_given_sample_size(341, 0.05, 0.10), 3)
    0.801
    """
    _validate_rates(p1, p2)
    _validate_prob("alpha", alpha)
    if n1 <= 0 or (n2 is not None and n2 <= 0):
        raise ValueError("sample sizes must be positive")
    n2 = n1 if n2 is None else n2

    z_a = _z_alpha(alpha, test_type, distribution, n1 + n2 - 2)

    if p1_observed_var is not None or use_pooled_variance:
        var = (
            p1_observed_var
            if p1_observed_var is not None
            else (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        )
        z_stat = abs(p2 - p1) / np.sqrt(var * (1 / n1 + 1 / n2))
    else:
        z_stat = abs(_cohens_h(p1, p2)) / np.sqrt(1 / n1 + 1 / n2)

    power = norm.cdf(z_stat - z_a)
    if test_type == "two_tail":
        power += norm.cdf(-z_stat - z_a)
    return float(power)


# ---------------------------------------------------------------------------
# Minimum detectable effect given sample size
# ---------------------------------------------------------------------------
def minimum_detectable_effect(
    n1: int,
    p1: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n2: Optional[int] = None,
    test_type: TestType = "one_tail",
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
    distribution: Distribution = "normal",
) -> tuple[float, float]:
    """
    Smallest upward effect detectable with the given sizes and power.

    Returns
    -------
    (mde_relative, p2_min)
        e.g. (0.20, 0.12) means a 20% relative lift, from p1 to 0.12.

    Notes
    -----
    With pooled variance the variance depends on p2, so it is solved by
    fixed-point iteration (converges in a handful of steps).
    """
    _validate_rates(p1)
    _validate_prob("alpha", alpha)
    _validate_prob("power", power)
    if n1 <= 0 or (n2 is not None and n2 <= 0):
        raise ValueError("sample sizes must be positive")
    n2 = n1 if n2 is None else n2

    z = _z_alpha(alpha, test_type, distribution, n1 + n2 - 2) + norm.ppf(power)
    inv_n = 1 / n1 + 1 / n2

    if p1_observed_var is not None:
        p2_min = p1 + z * np.sqrt(p1_observed_var * inv_n)
    elif use_pooled_variance:
        p2_min = p1 + z * np.sqrt(p1 * (1 - p1) * inv_n)  # starting point
        for _ in range(100):
            p2c = min(p2_min, 1 - 1e-12)
            var = (p1 * (1 - p1) + p2c * (1 - p2c)) / 2
            new = p1 + z * np.sqrt(var * inv_n)
            if abs(new - p2_min) < 1e-12:
                break
            p2_min = new
    else:
        h = z * np.sqrt(inv_n)
        a = np.arcsin(np.sqrt(p1)) + h / 2
        p2_min = 1.0 if a >= np.pi / 2 else np.sin(a) ** 2

    p2_min = float(min(p2_min, 1.0))
    return (p2_min - p1) / p1, p2_min


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
def estimate_test_duration(
    daily_users: int,
    sample_size_per_group: int,
    traffic_split: float = 1.0,
    n_groups: int = 2,
) -> float:
    """
    Days needed to collect `sample_size_per_group` users in each of `n_groups`.

    traffic_split: fraction of eligible daily users that enter the test
    (then split evenly across groups).

    >>> estimate_test_duration(10_000, 3835, traffic_split=0.5)
    1.534
    """
    if daily_users <= 0 or sample_size_per_group <= 0:
        raise ValueError("daily_users and sample_size_per_group must be positive")
    if not 0 < traffic_split <= 1:
        raise ValueError("traffic_split must be in (0, 1]")
    return sample_size_per_group * n_groups / (daily_users * traffic_split)


def estimate_test_duration_from_mde(
    daily_users: int,
    baseline_conv: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    traffic_split: float = 1.0,
    ratio: float = 1.0,
    test_type: TestType = "one_tail",
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
) -> dict:
    """
    Test duration needed to detect a relative lift `mde` (e.g. 0.1 = +10%).
    """
    if daily_users <= 0:
        raise ValueError("daily_users must be positive")
    if mde <= 0:
        raise ValueError("mde must be positive")
    if not 0 < traffic_split <= 1:
        raise ValueError("traffic_split must be in (0, 1]")
    target = baseline_conv * (1 + mde)
    if target >= 1:
        raise ValueError("Target conversion >= 100%; reduce mde or baseline_conv")

    n1, n2 = calculate_sample_size_given_power(
        baseline_conv,
        target,
        alpha,
        power,
        ratio,
        test_type,
        p1_observed_var,
        use_pooled_variance,
    )
    daily_test_users = daily_users * traffic_split
    days = (n1 + n2) / daily_test_users

    return {
        "duration_days": days,
        "duration_weeks": days / 7,
        "sample_size_control": n1,
        "sample_size_treatment": n2,
        "total_users_needed": n1 + n2,
        "daily_test_users": daily_test_users,
        "baseline_conversion": baseline_conv,
        "target_conversion": target,
        "mde_relative": mde,
        "mde_absolute": target - baseline_conv,
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
    }


def power_duration_tradeoff_analysis(
    daily_users: int,
    baseline_conv: float,
    target_conv: float,
    max_duration_days: int = 30,
    traffic_split: float = 1.0,
    alpha: float = 0.05,
    test_type: TestType = "one_tail",
    mde_power: float = 0.8,
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
) -> dict:
    """
    With a fixed maximum duration (equal groups), answer two questions:
      1. What power do I get for my target effect?
      2. What is the smallest lift I can detect with `mde_power` (default 80%)?
    """
    _validate_rates(baseline_conv, target_conv)
    if target_conv <= baseline_conv:
        raise ValueError("target_conv must be greater than baseline_conv")
    _validate_prob("mde_power", mde_power)

    n = int(daily_users * traffic_split * max_duration_days / 2)  # per group
    if n <= 0:
        raise ValueError("Duration constraint results in zero sample size")

    kw = dict(
        test_type=test_type,
        p1_observed_var=p1_observed_var,
        use_pooled_variance=use_pooled_variance,
    )
    achievable_power = calculate_power_given_sample_size(
        n, baseline_conv, target_conv, alpha, n, **kw
    )
    mde, min_conv = minimum_detectable_effect(
        n, baseline_conv, alpha, mde_power, n, **kw
    )
    target_effect = (target_conv - baseline_conv) / baseline_conv

    return {
        "sample_size_per_group": n,
        "duration_days": max_duration_days,
        "duration_weeks": max_duration_days / 7,
        "target_effect": target_effect,
        "achievable_power": achievable_power,
        "mde": mde,
        "mde_power": mde_power,
        "min_detectable_conv": min_conv,
        "target_detectable": target_effect >= mde,
        "recommendation": (
            "sufficient_power"
            if achievable_power >= mde_power
            else "insufficient_power"
        ),
    }


def calculate_power_over_time(
    daily_users: int,
    baseline_conv: float,
    target_conv: float,
    max_days: int = 60,
    traffic_split: float = 1.0,
    alpha: float = 0.05,
    test_type: TestType = "one_tail",
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
) -> tuple[list[int], list[float]]:
    """
    Power after each day of the test (equal groups), for a power curve.
    """
    _validate_rates(baseline_conv, target_conv)
    if target_conv <= baseline_conv:
        raise ValueError("target_conv must be greater than baseline_conv")

    days = list(range(1, max_days + 1))
    per_group_per_day = daily_users * traffic_split / 2
    powers = []
    for d in days:
        n = int(per_group_per_day * d)
        powers.append(
            calculate_power_given_sample_size(
                n,
                baseline_conv,
                target_conv,
                alpha,
                n,
                test_type,
                p1_observed_var,
                use_pooled_variance,
            )
            if n > 0
            else 0.0
        )
    return days, powers


def find_days_for_target_power(
    daily_users: int,
    baseline_conv: float,
    target_conv: float,
    target_power: float = 0.8,
    traffic_split: float = 1.0,
    alpha: float = 0.05,
    test_type: TestType = "one_tail",
    max_search_days: int = 365,
    p1_observed_var: Optional[float] = None,
    use_pooled_variance: bool = True,
) -> Optional[int]:
    """First whole day on which power >= target_power, or None."""
    days, powers = calculate_power_over_time(
        daily_users,
        baseline_conv,
        target_conv,
        max_search_days,
        traffic_split,
        alpha,
        test_type,
        p1_observed_var,
        use_pooled_variance,
    )
    return next((d for d, p in zip(days, powers) if p >= target_power), None)
