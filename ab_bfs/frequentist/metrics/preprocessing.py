"""Data preparation used before the frequentist evaluations."""

from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["winsorize", "cuped_adjustment", "prepare_data_for_analysis"]


def winsorize(
    df: pd.DataFrame,
    target_col: str,
    winsorization_method: Literal["pooled", "group"] = "pooled",
    group_col: str = "group",
    lower_percentile: float = 0.0,
    upper_percentile: float = 1.0,
) -> pd.Series:
    """
    Cap ``target_col`` at the given quantiles.

    Parameters
    ----------
    winsorization_method :
        "pooled" (default) computes one pair of caps on all groups together, so every
        group is treated by the same rule and the cap cannot absorb part of the
        treatment effect. "group" computes caps within each group.
    lower_percentile, upper_percentile : quantiles in [0, 1]; (0, 1) is a no-op.

    Returns
    -------
    pd.Series aligned with ``df``.
    """
    if not 0 <= lower_percentile < upper_percentile <= 1:
        raise ValueError("Need 0 <= lower_percentile < upper_percentile <= 1")
    s = df[target_col].astype(float)
    if winsorization_method == "pooled":
        lo, hi = s.quantile([lower_percentile, upper_percentile])
        return s.clip(lo, hi)
    if winsorization_method == "group":
        lo = df.groupby(group_col)[target_col].transform(lambda x: x.quantile(lower_percentile))
        hi = df.groupby(group_col)[target_col].transform(lambda x: x.quantile(upper_percentile))
        return s.clip(lo, hi)
    raise ValueError("winsorization_method must be 'pooled' or 'group'")


def cuped_adjustment(df: pd.DataFrame, target_col: str, covariate_col: str) -> pd.Series:
    """
    CUPED variance reduction: Y_adj = Y - theta (X - mean(X)), theta = Cov(Y, X) / Var(X).

    theta and mean(X) are computed on all rows (all groups pooled), which keeps the
    treatment effect unbiased. X must be measured before randomisation.
    """
    y = df[target_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        raise ValueError(f"Covariate '{covariate_col}' has zero variance")
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    return pd.Series(y - theta * (x - x.mean()), index=df.index, name=f"{target_col}_cuped")


def prepare_data_for_analysis(
    df: pd.DataFrame,
    temporal_col: str = "date",
    user_id_col: str = "user_id",
    metric_col: str = "conversion",
    group_col: str = "group",
    unit_time_col: Literal["s", "m", "h", "d"] = "d",
    aggregation_method: Literal["sum", "mean", "max"] = "sum",
) -> pd.DataFrame:
    """
    Bucket timestamps into integer time units (``time_int``, from the first event)
    and aggregate the metric per user, time unit and group.

    Use "max" for 0/1 conversions (did the user convert in that bucket), "sum" for
    revenue, "mean" for per-event averages.
    """
    out = df.copy()
    ts = pd.to_datetime(out[temporal_col])
    seconds = (ts - ts.min()).dt.total_seconds()
    divisor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit_time_col]
    out["time_int"] = (seconds // divisor).astype(int)
    return out.groupby([user_id_col, "time_int", group_col], observed=True).agg({metric_col: aggregation_method}).reset_index()
