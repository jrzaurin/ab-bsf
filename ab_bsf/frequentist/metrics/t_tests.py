"""
Evaluation of finished experiments with t/z tests.

Every function compares each non-reference group with ``reference_group`` and
returns ``{"<group>_vs_<reference>": {metric: value, ...}, ...}``.

All tests are Welch (unequal variances). ``alternative`` follows SciPy:
"two-sided", "larger" (treatment > control) or "smaller". Confidence intervals
are always two-sided at ``confidence_level``.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ab_bsf.frequentist.metrics.preprocessing import cuped_adjustment, winsorize

__all__ = [
    "t_test",
    "t_test_delta_lognormal",
    "t_test_did",
    "t_test_did_log",
    "t_test_did_regr",
]

Alternative = Literal["two-sided", "larger", "smaller"]
Results = Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_groups(
    df: pd.DataFrame, group_col: str, reference_group: Optional[str]
) -> tuple:
    groups = sorted(df[group_col].unique())
    if reference_group is None:
        reference_group = "control" if "control" in groups else groups[0]
    if reference_group not in groups:
        raise ValueError(
            f"reference_group '{reference_group}' not found in groups: {groups}"
        )
    return reference_group, [g for g in groups if g != reference_group]


def _winsorize_cols(dfc, cols, method, group_col, lower, upper):
    if lower > 0.0 or upper < 1.0:
        for col in cols:
            dfc[col] = winsorize(dfc, col, method, group_col, lower, upper)
    return dfc


def _p_value(t_stat: float, dof: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2 * stats.t.sf(abs(t_stat), dof))
    if alternative == "larger":
        return float(stats.t.sf(t_stat, dof))
    if alternative == "smaller":
        return float(stats.t.cdf(t_stat, dof))
    raise ValueError("alternative must be 'two-sided', 'larger' or 'smaller'")


def _welch(
    xt: np.ndarray, xc: np.ndarray, confidence_level: float, alternative: Alternative
) -> Dict[str, float]:
    """Welch t-test of mean(xt) - mean(xc) with CI and Cohen's d."""
    nt, nc = len(xt), len(xc)
    if nt < 2 or nc < 2:
        raise ValueError("Each group needs at least two observations")
    mt, mc = xt.mean(), xc.mean()
    vt, vc = xt.var(ddof=1), xc.var(ddof=1)
    se = np.sqrt(vt / nt + vc / nc)
    dof = (
        (vt / nt + vc / nc) ** 2
        / ((vt / nt) ** 2 / (nt - 1) + (vc / nc) ** 2 / (nc - 1))
        if se > 0
        else nt + nc - 2
    )
    diff = mt - mc
    t_stat = diff / se if se > 0 else 0.0
    t_crit = stats.t.ppf(1 - (1 - confidence_level) / 2, dof)
    pooled_sd = np.sqrt(((nt - 1) * vt + (nc - 1) * vc) / (nt + nc - 2))
    return {
        "mean_t": float(mt),
        "mean_c": float(mc),
        "var_t": float(vt),
        "var_c": float(vc),
        "n_t": nt,
        "n_c": nc,
        "diff": float(diff),
        "se": float(se),
        "dof": float(dof),
        "t": float(t_stat),
        "p": _p_value(t_stat, dof, alternative),
        "t_crit": float(t_crit),
        "ci_lower": float(diff - t_crit * se),
        "ci_upper": float(diff + t_crit * se),
        "d": float(diff / pooled_sd) if pooled_sd > 0 else 0.0,
    }


def _relative(w: Dict[str, float]) -> Dict[str, float]:
    """Relative difference mean_t / mean_c - 1 with a delta-method CI (both means uncertain)."""
    mt, mc = w["mean_t"], w["mean_c"]
    if mc == 0:
        return {
            "relative_difference": float("inf"),
            "relative_ci_lower": float("nan"),
            "relative_ci_upper": float("nan"),
        }
    ratio = mt / mc
    rel_var = (w["var_t"] / (w["n_t"] * mt**2) if mt != 0 else 0.0) + w["var_c"] / (
        w["n_c"] * mc**2
    )
    se = (
        abs(ratio) * np.sqrt(rel_var)
        if mt != 0
        else np.sqrt(w["var_t"] / w["n_t"]) / abs(mc)
    )
    return {
        "relative_difference": float(ratio - 1),
        "relative_ci_lower": float(ratio - 1 - w["t_crit"] * se),
        "relative_ci_upper": float(ratio - 1 + w["t_crit"] * se),
    }


# ---------------------------------------------------------------------------
# post-period only
# ---------------------------------------------------------------------------
def t_test(
    df: pd.DataFrame,
    target_col: str = "conversion",
    group_col: str = "group",
    reference_group: Optional[str] = None,
    agg_col: Optional[str] = None,
    covariate_col: Optional[str] = None,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    winsorization_method: Literal["pooled", "group"] = "pooled",
    winsorize_lower: float = 0.0,
    winsorize_upper: float = 1.0,
) -> Results:
    """
    Welch t-test on a binary or continuous metric.

    Parameters
    ----------
    agg_col : optional column to average over before testing (e.g. "user_id" so
        each user counts once, or a day column to test daily means).
    covariate_col : pre-experiment covariate; if given, CUPED is applied first.
    winsorize_* : optional capping before the test (and before CUPED).

    Returns
    -------
    Per comparison: ``control_mean``, ``feature_mean``, ``mean_differences``,
    ``ci_lower``/``ci_upper``, ``relative_difference`` with its CI, ``t_statistics``,
    ``p_values``, ``effect_sizes`` (Cohen's d), ``n_control``, ``n_feature``.
    """
    dfc = df.copy()
    dfc = _winsorize_cols(
        dfc,
        [target_col],
        winsorization_method,
        group_col,
        winsorize_lower,
        winsorize_upper,
    )
    if covariate_col is not None:
        dfc[target_col] = cuped_adjustment(dfc, target_col, covariate_col)
    if agg_col is not None:
        dfc = (
            dfc.groupby([agg_col, group_col], observed=True)[target_col]
            .mean()
            .reset_index()
        )

    ref, others = _resolve_groups(dfc, group_col, reference_group)
    xc = dfc.loc[dfc[group_col] == ref, target_col].to_numpy(float)
    out: Results = {}
    for g in others:
        w = _welch(
            dfc.loc[dfc[group_col] == g, target_col].to_numpy(float),
            xc,
            confidence_level,
            alternative,
        )
        out[f"{g}_vs_{ref}"] = {
            "control_mean": w["mean_c"],
            "feature_mean": w["mean_t"],
            "mean_differences": w["diff"],
            "ci_lower": w["ci_lower"],
            "ci_upper": w["ci_upper"],
            **_relative(w),
            "t_statistics": w["t"],
            "p_values": w["p"],
            "effect_sizes": w["d"],
            "n_control": w["n_c"],
            "n_feature": w["n_t"],
        }
    return out


def t_test_delta_lognormal(
    df: pd.DataFrame,
    target_col: str = "revenue",
    group_col: str = "group",
    reference_group: Optional[str] = None,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    bonferroni: bool = True,
    min_positive: float = 0.0,
    winsorization_method: Literal["pooled", "group"] = "pooled",
    winsorize_lower: float = 0.0,
    winsorize_upper: float = 1.0,
) -> Results:
    """
    Two-part test for zero-inflated, lognormal-ish metrics (mirrors the Bayesian
    delta-lognormal):

    1. conversion: z-test of P(value > min_positive), pooled SE for the test and
       unpooled SE for the CI;
    2. magnitude: Welch t-test on log(value) among converters, reported as the
       ratio of geometric means, exp(diff) - 1.

    With ``bonferroni`` each part is judged at alpha / 2 (``significant_*`` flags;
    p-values are raw). ``min_positive`` treats tiny values (dust) as zeros.
    Also returns the overall Welch test on raw means for reference.
    """
    dfc = df.copy()
    dfc = _winsorize_cols(
        dfc,
        [target_col],
        winsorization_method,
        group_col,
        winsorize_lower,
        winsorize_upper,
    )
    if (dfc[target_col] < 0).any():
        raise ValueError("t_test_delta_lognormal needs non-negative values")
    alpha = 1 - confidence_level
    alpha_part = alpha / 2 if bonferroni else alpha
    ref, others = _resolve_groups(dfc, group_col, reference_group)

    def split(g):
        x = dfc.loc[dfc[group_col] == g, target_col].to_numpy(float)
        return x, x > min_positive

    xc, cc = split(ref)
    out: Results = {}
    for g in others:
        xt, ct = split(g)
        nt, nc = len(xt), len(xc)
        pt, pc = ct.mean(), cc.mean()

        # Part 1: conversion
        pool = (ct.sum() + cc.sum()) / (nt + nc)
        se_pool = np.sqrt(pool * (1 - pool) * (1 / nt + 1 / nc))
        z = (pt - pc) / se_pool if se_pool > 0 else 0.0
        p_conv = _p_value(z, np.inf, alternative)
        se_unp = np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Part 2: magnitude on log scale
        lt, lc = np.log(xt[ct]), np.log(xc[cc])
        if len(lt) >= 2 and len(lc) >= 2:
            w = _welch(lt, lc, confidence_level, alternative)
            mag = {
                "log_mean_difference": w["diff"],
                "relative_diff_magnitude": float(np.exp(w["diff"]) - 1),
                "magnitude_ci_lower": float(np.exp(w["ci_lower"]) - 1),
                "magnitude_ci_upper": float(np.exp(w["ci_upper"]) - 1),
                "t_statistic_magnitude": w["t"],
                "p_value_magnitude": w["p"],
            }
        else:
            mag = {
                k: float("nan")
                for k in (
                    "log_mean_difference",
                    "relative_diff_magnitude",
                    "magnitude_ci_lower",
                    "magnitude_ci_upper",
                    "t_statistic_magnitude",
                )
            }
            mag["p_value_magnitude"] = 1.0

        overall = _welch(xt, xc, confidence_level, alternative)
        sig_conv = p_conv < alpha_part
        sig_mag = mag["p_value_magnitude"] < alpha_part
        out[f"{g}_vs_{ref}"] = {
            "n_control": nc,
            "n_feature": nt,
            "control_mean": overall["mean_c"],
            "feature_mean": overall["mean_t"],
            **{f"overall_{k}": v for k, v in _relative(overall).items()},
            "overall_p_value": overall["p"],
            "conversion_control": float(pc),
            "conversion_feature": float(pt),
            "conversion_difference": float(pt - pc),
            "conversion_ci_lower": float(pt - pc - z_crit * se_unp),
            "conversion_ci_upper": float(pt - pc + z_crit * se_unp),
            "relative_diff_conversion": float(pt / pc - 1) if pc > 0 else float("inf"),
            "z_statistic_conversion": float(z),
            "p_value_conversion": p_conv,
            "n_converters_control": int(cc.sum()),
            "n_converters_feature": int(ct.sum()),
            "geometric_mean_control": (
                float(np.exp(lc.mean())) if len(lc) else float("nan")
            ),
            "geometric_mean_feature": (
                float(np.exp(lt.mean())) if len(lt) else float("nan")
            ),
            **mag,
            "alpha_per_part": alpha_part,
            "significant_conversion": bool(sig_conv),
            "significant_magnitude": bool(sig_mag),
            "which_significant": (
                "both"
                if sig_conv and sig_mag
                else "conversion" if sig_conv else "magnitude" if sig_mag else "neither"
            ),
        }
    return out


# ---------------------------------------------------------------------------
# difference-in-differences (same units observed pre and post)
# ---------------------------------------------------------------------------
def _did_core(
    dfc,
    post_col,
    pre_col,
    group_col,
    ref,
    others,
    confidence_level,
    alternative,
    log_space,
):
    def arrays(g):
        sub = dfc[dfc[group_col] == g]
        post, pre = sub[post_col].to_numpy(float), sub[pre_col].to_numpy(float)
        return (np.log(post), np.log(pre)) if log_space else (post, pre)

    c_post, c_pre = arrays(ref)
    out: Results = {}
    for g in others:
        t_post, t_pre = arrays(g)
        w = _welch(t_post - t_pre, c_post - c_pre, confidence_level, alternative)
        res = {
            "did_estimates": w["diff"],
            "ci_lower": w["ci_lower"],
            "ci_upper": w["ci_upper"],
            "t_statistics": w["t"],
            "p_values": w["p"],
            "effect_sizes": w["d"],
            "post_differences": float(t_post.mean() - c_post.mean()),
            "pre_differences": float(t_pre.mean() - c_pre.mean()),
            "n_control": len(c_post),
            "n_feature": len(t_post),
        }
        means = {
            "control_pre_mean": c_pre.mean(),
            "control_post_mean": c_post.mean(),
            "feature_pre_mean": t_pre.mean(),
            "feature_post_mean": t_post.mean(),
        }
        if log_space:
            res.update(
                {k: float(np.exp(v)) for k, v in means.items()}
            )  # geometric means
            res.update(
                {
                    "relative_did": float(np.exp(w["diff"]) - 1),
                    "relative_ci_lower": float(np.exp(w["ci_lower"]) - 1),
                    "relative_ci_upper": float(np.exp(w["ci_upper"]) - 1),
                }
            )
        else:
            res.update({k: float(v) for k, v in means.items()})
            counterfactual = means["feature_pre_mean"] + (
                means["control_post_mean"] - means["control_pre_mean"]
            )
            if counterfactual != 0:
                res.update(
                    {
                        "relative_did": float(w["diff"] / counterfactual),
                        "relative_ci_lower": float(w["ci_lower"] / counterfactual),
                        "relative_ci_upper": float(w["ci_upper"] / counterfactual),
                    }
                )
            else:
                res.update(
                    {
                        "relative_did": float("nan"),
                        "relative_ci_lower": float("nan"),
                        "relative_ci_upper": float("nan"),
                    }
                )
        out[f"{g}_vs_{ref}"] = res
    return out


def t_test_did(
    df: pd.DataFrame,
    target_col: str,
    target_pre_col: str,
    group_col: str = "group",
    reference_group: Optional[str] = None,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    winsorization_method: Literal["pooled", "group"] = "pooled",
    winsorize_lower: float = 0.0,
    winsorize_upper: float = 1.0,
) -> Results:
    """
    Difference-in-differences as a Welch t-test on per-unit changes (post - pre).
    Each row is one unit observed in both periods.

    ``relative_did`` is the DiD over the treatment's counterfactual post mean,
    feature_pre_mean + (control_post_mean - control_pre_mean); its CI scales the
    DiD CI by that level and ignores the level's own uncertainty.
    """
    dfc = _winsorize_cols(
        df.copy(),
        [target_col, target_pre_col],
        winsorization_method,
        group_col,
        winsorize_lower,
        winsorize_upper,
    )
    ref, others = _resolve_groups(dfc, group_col, reference_group)
    return _did_core(
        dfc,
        target_col,
        target_pre_col,
        group_col,
        ref,
        others,
        confidence_level,
        alternative,
        False,
    )


def t_test_did_log(
    df: pd.DataFrame,
    target_col: str,
    target_pre_col: str,
    group_col: str = "group",
    reference_group: Optional[str] = None,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    winsorization_method: Literal["pooled", "group"] = "pooled",
    winsorize_lower: float = 0.0,
    winsorize_upper: float = 1.0,
) -> Results:
    """
    DiD in log space: Welch t-test on per-unit log(post / pre).

    ``did_estimates`` and its CI are log ratios; ``relative_did`` = exp(did) - 1 is
    the multiplicative effect on the *geometric* mean, and the reported group means
    are geometric means. Values must be strictly positive (winsorizing happens on the
    raw values, before the log).
    """
    dfc = _winsorize_cols(
        df.copy(),
        [target_col, target_pre_col],
        winsorization_method,
        group_col,
        winsorize_lower,
        winsorize_upper,
    )
    if (dfc[target_col] <= 0).any() or (dfc[target_pre_col] <= 0).any():
        raise ValueError(
            "t_test_did_log requires strictly positive values in both columns; "
            "filter them or use t_test_did on the raw scale."
        )
    ref, others = _resolve_groups(dfc, group_col, reference_group)
    return _did_core(
        dfc,
        target_col,
        target_pre_col,
        group_col,
        ref,
        others,
        confidence_level,
        alternative,
        True,
    )


def t_test_did_regr(
    df: pd.DataFrame,
    target_col: str,
    target_pre_col: str,
    group_col: str = "group",
    reference_group: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
    confidence_level: float = 0.95,
    winsorization_method: Literal["pooled", "group"] = "pooled",
    winsorize_lower: float = 0.0,
    winsorize_upper: float = 1.0,
) -> Results:
    """
    DiD via the interaction regression on the long (unit x period) panel:

        y_it = b0 + b1 treated_i + b2 post_t + b3 treated_i:post_t [+ covariates] + e_it

    ``b3`` is the DiD. Standard errors are clustered by unit: each unit contributes
    a pre and a post row, so treating rows as independent (plain HC3) would ignore
    the within-unit correlation and misstate the SE. Without covariates the estimate
    equals ``t_test_did``'s and the SE is asymptotically the same.

    ``covariate_cols`` are unit-level columns added as additive terms.
    """
    covariate_cols = list(covariate_cols or [])
    dfc = _winsorize_cols(
        df.copy(),
        [target_col, target_pre_col],
        winsorization_method,
        group_col,
        winsorize_lower,
        winsorize_upper,
    )
    ref, others = _resolve_groups(dfc, group_col, reference_group)
    dfc = dfc.reset_index(drop=True)
    dfc["_unit"] = np.arange(len(dfc))
    keep = ["_unit", group_col] + covariate_cols
    pre = (
        dfc[keep + [target_pre_col]]
        .rename(columns={target_pre_col: "_y"})
        .assign(period=0)
    )
    post = dfc[keep + [target_col]].rename(columns={target_col: "_y"}).assign(period=1)
    long = pd.concat([pre, post], ignore_index=True)
    alpha = 1 - confidence_level

    out: Results = {}
    for g in others:
        sub = long[long[group_col].isin([ref, g])].copy()
        sub["treated"] = (sub[group_col] == g).astype(int)
        formula = "_y ~ treated + period + treated:period" + "".join(
            f" + {c}" for c in covariate_cols
        )
        model = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["_unit"]}
        )
        b3 = "treated:period"
        lo, hi = model.conf_int(alpha=alpha).loc[b3]
        means = sub.groupby(["treated", "period"])["_y"].mean()
        c_pre, c_post, t_pre, t_post = (
            means[(0, 0)],
            means[(0, 1)],
            means[(1, 0)],
            means[(1, 1)],
        )
        counterfactual = t_pre + (c_post - c_pre)
        did = float(model.params[b3])
        out[f"{g}_vs_{ref}"] = {
            "did_estimates": did,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "t_statistics": float(model.tvalues[b3]),
            "p_values": float(model.pvalues[b3]),
            "std_error": float(model.bse[b3]),
            "effect_sizes": (
                did / float(np.sqrt(model.mse_resid)) if model.mse_resid > 0 else 0.0
            ),
            "post_differences": float(t_post - c_post),
            "pre_differences": float(t_pre - c_pre),
            "control_pre_mean": float(c_pre),
            "control_post_mean": float(c_post),
            "feature_pre_mean": float(t_pre),
            "feature_post_mean": float(t_post),
            "relative_did": did / counterfactual if counterfactual else float("nan"),
            "relative_ci_lower": (
                float(lo) / counterfactual if counterfactual else float("nan")
            ),
            "relative_ci_upper": (
                float(hi) / counterfactual if counterfactual else float("nan")
            ),
        }
    return out
