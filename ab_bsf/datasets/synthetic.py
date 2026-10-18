"""
Synthetic experiment data for examples and tests.

(The original package shipped a ``session_data.parquet`` loaded with
``importlib.resources``; that file could not be recovered, so these generators
take its place.)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["make_conversion_data", "make_revenue_data", "make_did_data"]


def make_conversion_data(
    n_days: int = 28,
    base_rate: float = 0.12,
    lifts: Optional[Dict[str, float]] = None,
    users_per_day: int = 1000,
    day_sd: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """
    One row per user: ``day``, ``group``, ``conversion`` (0/1).

    ``lifts`` maps group -> relative lift in the odds (control is added with 0).
    Day effects (weekly cycle + noise) act on the logit scale and are shared by
    all groups, so fixed/mixed time effects have something to absorb.
    """
    lifts = {"treatment": 0.10} if lifts is None else lifts
    groups = {"control": 0.0, **lifts}
    rng = np.random.default_rng(seed)
    base = np.log(base_rate / (1 - base_rate))
    frames = []
    for d in range(n_days):
        shock = 0.25 * np.sin(2 * np.pi * d / 7) + rng.normal(0, day_sd)
        for g, lift in groups.items():
            p = 1 / (1 + np.exp(-(base + shock + np.log1p(lift))))
            frames.append(
                pd.DataFrame(
                    {
                        "day": d,
                        "group": g,
                        "conversion": rng.binomial(1, p, users_per_day),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def make_revenue_data(
    n_per_group: int = 5000,
    conversion_rate: float = 0.3,
    log_mean: float = 3.0,
    log_sd: float = 1.0,
    conversion_lift: float = 0.0,
    magnitude_lift: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Zero-inflated lognormal revenue: ``group`` (control/treatment) and ``revenue``.
    Treatment multiplies the conversion rate by (1 + conversion_lift) and the
    lognormal median by (1 + magnitude_lift).
    """
    rng = np.random.default_rng(seed)
    out = []
    for g, cl, ml in [
        ("control", 0.0, 0.0),
        ("treatment", conversion_lift, magnitude_lift),
    ]:
        conv = rng.random(n_per_group) < conversion_rate * (1 + cl)
        rev = np.where(
            conv, rng.lognormal(log_mean + np.log1p(ml), log_sd, n_per_group), 0.0
        )
        out.append(pd.DataFrame({"group": g, "revenue": rev}))
    return pd.concat(out, ignore_index=True)


def make_did_data(
    n_per_group: int = 2000,
    control_pre_mean: float = 50.0,
    baseline_gap: float = 2.0,
    common_trend: float = 5.0,
    effect: float = 3.0,
    unit_sd: float = 10.0,
    noise_sd: float = 5.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Wide panel, one row per user: ``group``, ``pre``, ``post``.
    Treatment starts ``baseline_gap`` higher; both groups drift by ``common_trend``;
    treatment adds ``effect``. Users have persistent levels (``unit_sd``), so pre and
    post are correlated, as in real pre/post data.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g, gap, eff in [("control", 0.0, 0.0), ("treatment", baseline_gap, effect)]:
        level = rng.normal(control_pre_mean + gap, unit_sd, n_per_group)
        pre = level + rng.normal(0, noise_sd, n_per_group)
        post = level + common_trend + eff + rng.normal(0, noise_sd, n_per_group)
        out.append(pd.DataFrame({"group": g, "pre": pre, "post": post}))
    return pd.concat(out, ignore_index=True)
