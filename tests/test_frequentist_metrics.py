import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from ab_bsf.datasets import make_conversion_data, make_did_data, make_revenue_data
from ab_bsf.frequentist.metrics import (
    adjust_p_values,
    cuped_adjustment,
    fixed_effects_linear_regression,
    fixed_effects_logistic_regression,
    mixed_effects_linear_regression,
    prepare_data_for_analysis,
    summarise_treatment_effects,
    t_test,
    t_test_delta_lognormal,
    t_test_did,
    t_test_did_log,
    t_test_did_regr,
    winsorize,
)


def two_groups(xc, xt):
    return pd.DataFrame({"group": ["control"] * len(xc) + ["treatment"] * len(xt), "y": np.r_[xc, xt]})


def test_t_test_matches_scipy_welch():
    rng = np.random.default_rng(0)
    xc, xt = rng.normal(10, 2, 300), rng.normal(10.5, 4, 500)
    r = t_test(two_groups(xc, xt), "y")["treatment_vs_control"]
    ref = stats.ttest_ind(xt, xc, equal_var=False)
    assert r["t_statistics"] == pytest.approx(ref.statistic)
    assert r["p_values"] == pytest.approx(ref.pvalue)
    ci = ref.confidence_interval(0.95)
    assert (r["ci_lower"], r["ci_upper"]) == pytest.approx((ci.low, ci.high))
    one = t_test(two_groups(xc, xt), "y", alternative="larger")["treatment_vs_control"]["p_values"]
    assert one == pytest.approx(stats.ttest_ind(xt, xc, equal_var=False, alternative="greater").pvalue)


def test_relative_ci_coverage():
    rng = np.random.default_rng(1)
    hits, reps = 0, 500
    for _ in range(reps):
        xc, xt = rng.exponential(10, 400), rng.exponential(11, 400)
        r = t_test(two_groups(xc, xt), "y")["treatment_vs_control"]
        hits += r["relative_ci_lower"] <= 0.1 <= r["relative_ci_upper"]
    assert 0.92 <= hits / reps <= 0.975


def test_agg_col_and_multiple_arms():
    df = make_conversion_data(lifts={"b": 0.1, "c": 0.0}, n_days=7, users_per_day=200)
    daily = t_test(df, "conversion", agg_col="day")
    assert set(daily) == {"b_vs_control", "c_vs_control"}
    assert daily["b_vs_control"]["n_control"] == 7
    adj = adjust_p_values(daily, method="holm")
    assert all(adj[k]["p_values_adjusted"] >= daily[k]["p_values"] for k in daily)


def test_winsorize_pooled_vs_group():
    df = two_groups(np.arange(100.0), np.arange(100.0) * 10)
    pooled = winsorize(df, "y", "pooled", upper_percentile=0.9)
    group = winsorize(df, "y", "group", upper_percentile=0.9)
    assert pooled.max() == pytest.approx(np.quantile(df["y"], 0.9))
    assert group[df.group == "control"].max() == pytest.approx(np.quantile(np.arange(100.0), 0.9))


def test_cuped_preserves_effect_and_reduces_variance():
    rng = np.random.default_rng(2)
    diffs_raw, diffs_cuped = [], []
    for _ in range(300):
        pre = rng.normal(50, 15, 2000)
        post = 0.8 * pre + rng.normal(0, 6, 2000)
        g = np.repeat(["control", "treatment"], 1000)
        post[g == "treatment"] += 2
        df = pd.DataFrame({"group": g, "pre": pre, "post": post})
        diffs_raw.append(t_test(df, "post")["treatment_vs_control"]["mean_differences"])
        diffs_cuped.append(t_test(df, "post", covariate_col="pre")["treatment_vs_control"]["mean_differences"])
    assert np.mean(diffs_cuped) == pytest.approx(2, abs=0.05)
    assert np.var(diffs_cuped) < 0.3 * np.var(diffs_raw)
    df = pd.DataFrame({"pre": [1.0, 1.0], "post": [1.0, 2.0]})
    with pytest.raises(ValueError):
        cuped_adjustment(df, "post", "pre")


def test_delta_lognormal_parts_and_type_one_error():
    df = make_revenue_data(n_per_group=20_000, conversion_lift=0.0, magnitude_lift=0.2, seed=3)
    r = t_test_delta_lognormal(df, "revenue")["treatment_vs_control"]
    assert r["relative_diff_magnitude"] == pytest.approx(0.2, abs=0.05)
    assert r["significant_magnitude"] and not r["significant_conversion"]
    rejections = 0
    for s in range(300):
        d0 = make_revenue_data(n_per_group=2000, seed=100 + s)
        rejections += t_test_delta_lognormal(d0, "revenue")["treatment_vs_control"]["which_significant"] != "neither"
    assert rejections / 300 < 0.08  # nominal ~0.05 with Bonferroni over two parts


def test_did_family_agrees():
    df = make_did_data(n_per_group=3000, effect=3.0, seed=4)
    a = t_test_did(df, "post", "pre")["treatment_vs_control"]
    manual = df[df.group == "treatment"].eval("post - pre").mean() - df[df.group == "control"].eval("post - pre").mean()
    assert a["did_estimates"] == pytest.approx(manual)
    assert a["did_estimates"] == pytest.approx(3, abs=0.5)
    b = t_test_did_regr(df, "post", "pre")["treatment_vs_control"]
    assert b["did_estimates"] == pytest.approx(a["did_estimates"])
    se_a = (a["ci_upper"] - a["ci_lower"]) / (2 * stats.t.ppf(0.975, 6000))
    assert b["std_error"] == pytest.approx(se_a, rel=0.02)
    # naive row-level HC3 would overstate the SE because pre/post are correlated
    long = pd.concat([df.assign(y=df.pre, period=0), df.assign(y=df.post, period=1)])
    long["treated"] = (long.group == "treatment").astype(int)
    naive = smf.ols("y ~ treated * period", long).fit(cov_type="HC3").bse["treated:period"]
    assert naive > 1.5 * b["std_error"]


def test_did_log():
    rng = np.random.default_rng(5)
    n = 3000
    lvl = rng.normal(3, 1, 2 * n)
    pre = np.exp(lvl + rng.normal(0, 0.3, 2 * n))
    g = np.repeat(["control", "treatment"], n)
    post = np.exp(lvl + 0.05 + np.where(g == "treatment", np.log(1.1), 0) + rng.normal(0, 0.3, 2 * n))
    r = t_test_did_log(pd.DataFrame({"group": g, "pre": pre, "post": post}), "post", "pre")["treatment_vs_control"]
    assert r["relative_did"] == pytest.approx(0.10, abs=0.03)
    with pytest.raises(ValueError):
        t_test_did_log(pd.DataFrame({"group": g, "pre": pre * 0, "post": post}), "post", "pre")


def test_regressions_fixed_vs_mixed():
    df = make_conversion_data(lifts={"treatment": 0.10}, n_days=14, users_per_day=2000, seed=6)
    fe = summarise_treatment_effects(fixed_effects_logistic_regression(df, "day"))
    row = smf.logit("conversion ~ C(group, Treatment('control')) + C(day)", df).fit(disp=0)
    assert fe["treatment"]["estimate"] == pytest.approx(row.params.iloc[1], rel=1e-6)
    assert fe["treatment"]["odds_ratio"] == pytest.approx(1.10, abs=0.06)
    me = summarise_treatment_effects(mixed_effects_linear_regression(df, "day", "conversion"))
    ols = summarise_treatment_effects(fixed_effects_linear_regression(df, "day", "conversion"))
    assert me["treatment"]["estimate"] == pytest.approx(ols["treatment"]["estimate"], abs=0.002)


def test_prepare_data_for_analysis():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 12:00", "2025-01-02 11:00"]),
            "user_id": [1, 1, 1],
            "group": ["control"] * 3,
            "conversion": [0, 1, 0],
        }
    )
    out = prepare_data_for_analysis(df, aggregation_method="max")
    assert out["time_int"].tolist() == [0, 1]  # buckets are 24h from the first event
    assert out.loc[out.time_int == 0, "conversion"].item() == 1
