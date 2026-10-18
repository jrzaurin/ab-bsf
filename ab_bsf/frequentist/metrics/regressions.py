"""
Regression-based evaluation with time-period effects (fixed vs mixed).

* Fixed effects: one dummy per time bucket, removes every bucket-specific shock.
  Best with few buckets that each have plenty of data (e.g. day-of-week).
* Mixed effects: bucket intercepts ~ N(0, s^2), partial pooling. Best with many
  buckets of uneven size. Assumes normal bucket effects; extreme buckets shrink.

Input is long data with one row per observation (or per user x bucket, see
``prepare_data_for_analysis``). Use ``summarise_treatment_effects`` to pull the
per-group effects out of any result.
"""

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

__all__ = [
    "fixed_effects_linear_regression",
    "mixed_effects_linear_regression",
    "fixed_effects_logistic_regression",
    "mixed_effects_logistic_regression",
    "summarise_treatment_effects",
]


def _group_term(group_col: str, reference: str) -> str:
    return f"C({group_col}, Treatment('{reference}'))"


def _reference(df: pd.DataFrame, group_col: str, reference_group: Optional[str]) -> str:
    groups = sorted(df[group_col].unique())
    ref = reference_group or ("control" if "control" in groups else groups[0])
    if ref not in groups:
        raise ValueError(f"reference_group '{ref}' not found in {groups}")
    return ref


def fixed_effects_linear_regression(
    df: pd.DataFrame,
    temporal_col: str = "time_int",
    metric_col: str = "revenue",
    group_col: str = "group",
    reference_group: Optional[str] = None,
    cluster_col: Optional[str] = None,
):
    """
    OLS ``metric ~ group + C(time)``.

    ``cluster_col`` (e.g. the time bucket or user id) gives cluster-robust SEs;
    otherwise HC1 robust SEs. Returns a statsmodels results object.
    """
    ref = _reference(df, group_col, reference_group)
    formula = f"{metric_col} ~ {_group_term(group_col, ref)} + C({temporal_col})"
    model = smf.ols(formula, data=df)
    if cluster_col is not None:
        return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return model.fit(cov_type="HC1")


def mixed_effects_linear_regression(
    df: pd.DataFrame,
    temporal_col: str = "time_int",
    metric_col: str = "revenue",
    group_col: str = "group",
    reference_group: Optional[str] = None,
):
    """
    Linear mixed model ``metric ~ group + (1 | time)`` (REML).

    Also usable for 0/1 outcomes as a linear probability model, which with large
    samples gives reliable effects on the percentage-point scale.
    """
    ref = _reference(df, group_col, reference_group)
    model = smf.mixedlm(f"{metric_col} ~ {_group_term(group_col, ref)}", df, groups=df[temporal_col])
    return model.fit(reml=True)


def fixed_effects_logistic_regression(
    df: pd.DataFrame,
    temporal_col: str = "time_int",
    metric_col: str = "conversion",
    group_col: str = "group",
    reference_group: Optional[str] = None,
    cluster_by_time: bool = True,
):
    """
    Logit ``conversion ~ group + C(time)``, fitted as a binomial GLM on
    (time x group) counts: identical estimates to the per-row fit, much faster.
    SEs are clustered by time bucket unless ``cluster_by_time=False``.
    Coefficients are log odds ratios.
    """
    ref = _reference(df, group_col, reference_group)
    agg = df.groupby([temporal_col, group_col], observed=True)[metric_col].agg(_succ="sum", _n="count").reset_index()
    agg["_fail"] = agg["_n"] - agg["_succ"]
    formula = f"_succ + _fail ~ {_group_term(group_col, ref)} + C({temporal_col})"
    model = smf.glm(formula, agg, family=sm.families.Binomial())
    if cluster_by_time and agg[temporal_col].nunique() > 1:
        return model.fit(cov_type="cluster", cov_kwds={"groups": agg[temporal_col]})
    return model.fit()


def mixed_effects_logistic_regression(
    df: pd.DataFrame,
    temporal_col: str = "time_int",
    metric_col: str = "conversion",
    group_col: str = "group",
    reference_group: Optional[str] = None,
):
    """
    Logistic GLMM ``logit(p) ~ group + (1 | time)``.

    statsmodels has no frequentist logistic GLMM, so this uses
    ``BinomialBayesMixedGLM.fit_vb`` (variational Bayes): the result has posterior
    means/SDs (``fe_mean``, ``fe_sd``), not p-values, and VB tends to understate
    uncertainty. For a frequentist GLMM use lme4 (R) or pymer4; for p-values in
    Python, use ``mixed_effects_linear_regression`` on the 0/1 outcome.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    ref = _reference(df, group_col, reference_group)
    model = BinomialBayesMixedGLM.from_formula(
        f"{metric_col} ~ {_group_term(group_col, ref)}", {"time": f"0 + C({temporal_col})"}, df
    )
    result = model.fit_vb()
    result._pump_exog_names = model.exog_names  # used by summarise_treatment_effects
    return result


def summarise_treatment_effects(
    result, group_col: str = "group", confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Pull ``{"<group>": {"estimate", "ci_lower", "ci_upper", "p_value"}}`` out of any
    result returned above. For logistic models the estimate is a log odds ratio and
    ``odds_ratio`` is added. For the VB GLMM, ``p_value`` is replaced by
    ``prob_positive`` (posterior P(effect > 0)).
    """
    alpha = 1 - confidence_level
    out: Dict[str, Dict[str, float]] = {}
    is_logit = False

    if hasattr(result, "fe_mean"):  # BinomialBayesMixedGLM
        names = result._pump_exog_names
        z = stats.norm.ppf(1 - alpha / 2)
        for i, name in enumerate(names):
            if name.startswith(f"C({group_col}"):
                m, s = result.fe_mean[i], result.fe_sd[i]
                g = name.split("[T.")[-1].rstrip("]")
                out[g] = {
                    "estimate": float(m),
                    "ci_lower": float(m - z * s),
                    "ci_upper": float(m + z * s),
                    "prob_positive": float(stats.norm.sf(0, m, s)),
                    "odds_ratio": float(np.exp(m)),
                }
        return out

    ci = result.conf_int(alpha=alpha)
    family = getattr(getattr(result, "model", None), "family", None)
    is_logit = isinstance(family, sm.families.Binomial)
    for name in result.params.index:
        if name.startswith(f"C({group_col}"):
            g = name.split("[T.")[-1].rstrip("]")
            est = float(result.params[name])
            row = {
                "estimate": est,
                "ci_lower": float(ci.loc[name, 0]),
                "ci_upper": float(ci.loc[name, 1]),
                "p_value": float(result.pvalues[name]),
            }
            if is_logit:
                row["odds_ratio"] = float(np.exp(est))
            out[g] = row
    return out
