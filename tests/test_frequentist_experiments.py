import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from ab_bfs.frequentist.experiments import (
    BinaryFrequentistTest,
    ContinuousFrequentistTest,
    DeltaLognormalFrequentistTest,
    clustered_sample_size,
    design_effect,
)
from ab_bfs.frequentist.metrics import t_test_delta_lognormal
from ab_bfs.utils import alias_params

ALT = {"one_tail": "larger", "two_tail": "two-sided"}


# --------------------------------------------------------------------- binary
@pytest.mark.parametrize("p1,p2", [(0.10, 0.12), (0.05, 0.10), (0.03, 0.0345)])
@pytest.mark.parametrize("tt", ["one_tail", "two_tail"])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_binary_arcsine_matches_statsmodels(p1, p2, tt, ratio):
    t = BinaryFrequentistTest(n1=1000, p1=p1, test_type=tt, ratio=ratio, use_pooled_variance=False)
    n1, n2 = t.sample_size_given_power(p2)
    sm = NormalIndPower().solve_power(proportion_effectsize(p2, p1), alpha=0.05, power=0.8, ratio=ratio, alternative=ALT[tt])
    assert 0 <= n1 - sm < 2


@pytest.mark.parametrize("opts", [{}, {"use_pooled_variance": False}, {"p1_observed_var": 0.1 * 0.9}])
@pytest.mark.parametrize("tt", ["one_tail", "two_tail"])
def test_binary_round_trips(opts, tt):
    t = BinaryFrequentistTest(n1=1000, p1=0.1, test_type=tt, **opts)
    n1, _ = t.sample_size_given_power(0.12)
    assert 0.8 <= t.power_given_sample_size(0.12, n1) < 0.802
    _, p2 = t.minimum_detectable_effect(n1)
    assert abs(p2 - 0.12) / 0.02 < 0.01


def test_binary_power_by_simulation():
    t = BinaryFrequentistTest(n1=1000, p1=0.05)  # one tail, pooled
    n, _ = t.sample_size_given_power(0.10)
    rng = np.random.default_rng(0)
    R = 40_000
    x1, x2 = rng.binomial(n, 0.05, R), rng.binomial(n, 0.10, R)
    pp = (x1 + x2) / (2 * n)
    z = (x2 - x1) / n / np.sqrt(pp * (1 - pp) * 2 / n)
    assert (z > 1.6449).mean() == pytest.approx(0.8, abs=0.02)


def test_multivariant_bonferroni_and_duration():
    t = BinaryFrequentistTest(n1=1000, p1=0.1, n_variants=3)
    assert t.alpha_ == pytest.approx(0.025)
    n_c, n_ts = t.sample_size_given_power_multivariant(0.12)
    assert n_ts == [n_c, n_c]
    single = BinaryFrequentistTest(n1=1000, p1=0.1).sample_size_given_power(0.12)[0]
    assert n_c > single
    per, overall = t.power_given_sample_size_multivariant(0.12, n_c)
    assert overall > per >= 0.8
    with pytest.raises(ValueError):
        BinaryFrequentistTest(n1=1000, p1=0.1, n_variants=3, ratio=2.0)


def test_duration_never_negative_and_from_scratch():
    # regression test for the negative-days bug
    t = ContinuousFrequentistTest(n1=1000, mean1=100, std1=20)
    r = t.test_duration(daily_observations=500, mde=0.1)
    assert r["duration_days"] == 0.0 and r["required_control_sample_size"] == 50
    r0 = t.test_duration(daily_observations=10, mde=0.1, already_collected=False)
    assert r0["duration_days"] == 5.0
    day = t.days_for_target_power(10, 110)
    assert day == 5


def test_mde_given_duration_uses_fixed_power():
    t = BinaryFrequentistTest(n1=1000, p1=0.1)
    out = t.mde_given_duration(daily_observations=500, days=14)
    n = out["control_sample_size"]
    assert t.power_given_sample_size(out["target_mean"], n) == pytest.approx(0.8, abs=1e-6)


# ----------------------------------------------------------------- continuous
@pytest.mark.parametrize("ratio", [1.0, 0.5, 3.0])
def test_continuous_matches_statsmodels_normal(ratio):
    t = ContinuousFrequentistTest(n1=100, mean1=100, std1=20, ratio=ratio, test_type="two_tail")
    n1, _ = t.sample_size_given_power(105)
    sm = NormalIndPower().solve_power(5 / 20, alpha=0.05, power=0.8, ratio=ratio, alternative="two-sided")
    assert 0 <= n1 - sm < 2
    assert t.power_given_sample_size(105, n1) >= 0.8


def test_continuous_unequal_std():
    t = ContinuousFrequentistTest(n1=100, mean1=100, std1=10, std2=30)
    n1, n2 = t.sample_size_given_power(105)
    assert t.power_given_sample_size(105, n1) == pytest.approx(0.8, abs=0.005)


def test_alias_params():
    assert ContinuousFrequentistTest(n_control=10, mean1=5, std1=1).n1 == 10
    with pytest.raises(TypeError):
        ContinuousFrequentistTest(n1=10, p1=5, mean1=5, std1=1)

    @alias_params(x=["a", "b"])
    def f(x):
        return x

    assert f(b=2) == 2


# ------------------------------------------------------------ delta-lognormal
def test_delta_lognormal_validation():
    with pytest.raises(ValueError):
        DeltaLognormalFrequentistTest(n1=1000, p1=20, conversion_rate=0.3, mean_positive=35, std_log=0.5)
    t = DeltaLognormalFrequentistTest(n1=1000, p1=10.5, conversion_rate=0.3, mean_positive=35, std_positive=18.6)
    assert t.std_log == pytest.approx(0.5, abs=0.01)


def test_delta_lognormal_sample_size_achieves_power_by_simulation():
    cr, sigma, mu = 0.3, 0.5, np.log(35) - 0.125
    t = DeltaLognormalFrequentistTest(
        n1=1000, p1=0.3 * 35, conversion_rate=cr, mean_positive=35, std_log=sigma, test_type="two_tail"
    )
    target = t.p1 * 1.10
    n, _ = t.sample_size_given_power(target)
    conv2, mag2 = t._split(target)
    rng = np.random.default_rng(0)
    import pandas as pd

    hits = 0
    reps = 300
    for _ in range(reps):
        c = np.where(rng.random(n) < cr, rng.lognormal(mu, sigma, n), 0)
        tr = np.where(rng.random(n) < conv2, rng.lognormal(mu + np.log(mag2 / 35), sigma, n), 0)
        df = pd.DataFrame({"group": ["control"] * n + ["treatment"] * n, "revenue": np.r_[c, tr]})
        r = t_test_delta_lognormal(df, "revenue")["treatment_vs_control"]
        hits += r["which_significant"] != "neither"
    # both parts sized for 80% -> P(at least one) should be ~ 1 - 0.2^2 = 0.96
    assert hits / reps > 0.9


# ------------------------------------------------------------------ clustered
def test_design_effect():
    assert design_effect(20, 0.05) == pytest.approx(1.95)
    assert design_effect(20, 0.05, cluster_size_cv=0.5) > design_effect(20, 0.05)
    n_c, n_t, info = clustered_sample_size(1000, 1000, 20, 0.05)
    assert info["n_clusters_control"] == 98 and n_c == 1960
