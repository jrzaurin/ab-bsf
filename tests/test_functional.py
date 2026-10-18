"""Checks power_analysis.py against statsmodels, round trips and simulation."""

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from ab_bfs.frequentist.experiments.functional import (
    calculate_power_given_sample_size,
    calculate_sample_size_given_power,
    estimate_test_duration,
    estimate_test_duration_from_mde,
    find_days_for_target_power,
    minimum_detectable_effect,
    power_duration_tradeoff_analysis,
)

CASES = [(0.10, 0.12), (0.05, 0.10), (0.03, 0.0345), (0.30, 0.33)]
ALT = {"one_tail": "larger", "two_tail": "two-sided"}


@pytest.mark.parametrize("p1,p2", CASES)
@pytest.mark.parametrize("tt", ["one_tail", "two_tail"])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_arcsine_sample_size_matches_statsmodels(p1, p2, tt, ratio):
    n1, n2 = calculate_sample_size_given_power(p1, p2, test_type=tt, ratio=ratio, use_pooled_variance=False)
    sm = NormalIndPower().solve_power(proportion_effectsize(p2, p1), alpha=0.05, power=0.8, ratio=ratio, alternative=ALT[tt])
    assert 0 <= n1 - sm < 2  # ceil, plus opposite tail in two-sided
    assert n2 == int(np.ceil(n1 * ratio))


@pytest.mark.parametrize("p1,p2", CASES)
@pytest.mark.parametrize("tt", ["one_tail", "two_tail"])
@pytest.mark.parametrize("pooled", [True, False])
def test_round_trips(p1, p2, tt, pooled):
    n1, n2 = calculate_sample_size_given_power(p1, p2, test_type=tt, use_pooled_variance=pooled)
    pw = calculate_power_given_sample_size(n1, p1, p2, n2=n2, test_type=tt, use_pooled_variance=pooled)
    assert 0.8 <= pw < 0.802
    _, p2_min = minimum_detectable_effect(n1, p1, n2=n2, test_type=tt, use_pooled_variance=pooled)
    assert abs(p2_min - p2) / (p2 - p1) < 0.01  # within 1% of the effect


def test_observed_var_matches_pooled_when_equal():
    p1, p2 = 0.1, 0.12
    var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    assert calculate_sample_size_given_power(p1, p2, p1_observed_var=var) == calculate_sample_size_given_power(p1, p2)


@pytest.mark.parametrize("tt", ["one_tail", "two_tail"])
def test_power_matches_simulation(tt):
    p1, p2 = 0.05, 0.10
    n, _ = calculate_sample_size_given_power(p1, p2, test_type=tt)
    rng = np.random.default_rng(0)
    R = 40_000
    x1, x2 = rng.binomial(n, p1, R), rng.binomial(n, p2, R)
    pp = (x1 + x2) / (2 * n)
    z = (x2 - x1) / n / np.sqrt(pp * (1 - pp) * 2 / n)
    crit = 1.959964 if tt == "two_tail" else 1.644854
    sim = (np.abs(z) > crit).mean() if tt == "two_tail" else (z > crit).mean()
    assert abs(sim - 0.8) < 0.02


def test_duration_consistency():
    r = estimate_test_duration_from_mde(5_000, 0.10, 0.20, traffic_split=0.5)
    assert np.isclose(r["duration_days"], estimate_test_duration(5_000, r["sample_size_control"], 0.5))
    # whole-day search agrees with the continuous duration
    d = find_days_for_target_power(5_000, 0.10, 0.12, traffic_split=0.5)
    assert d == int(np.ceil(r["duration_days"])) or abs(d - r["duration_days"]) <= 1


def test_tradeoff_not_circular():
    r = power_duration_tradeoff_analysis(1_000, 0.10, 0.12, max_duration_days=14)
    _, p2_80 = minimum_detectable_effect(r["sample_size_per_group"], 0.10)
    assert np.isclose(r["min_detectable_conv"], p2_80)
    assert r["target_detectable"] == (r["target_effect"] >= r["mde"])
    assert r["target_detectable"] == (r["achievable_power"] >= 0.8)
