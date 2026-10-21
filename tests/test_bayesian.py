import numpy as np
import pytest
from scipy import integrate, stats

from ab_bsf.bayesian.experiments import (
    BinaryBayesianTest,
    DeltaLognormalBayesianTest,
    DeltaNormalBayesianTest,
    DiscreteBayesianTest,
    ExponentialBayesianTest,
    NormalBayesianTest,
    PoissonBayesianTest,
)
from ab_bsf.bayesian.models import (
    DeltaLognormal,
    DirichletMultinomial,
    NormalInverseGamma,
)


def prob_b_beats_a(a1, b1, a2, b2):
    """Exact P(theta_B > theta_A) for independent Betas by numerical integration."""
    f = lambda x: stats.beta.pdf(x, a2, b2) * stats.beta.cdf(x, a1, b1)  # noqa: E731
    return integrate.quad(f, 0, 1, limit=200)[0]


# --------------------------------------------------------------------- binary
def test_binary_matches_reference_readme():
    # numbers from the bayesian-testing README (same priors and data)
    t = BinaryBayesianTest()
    t.add_variant_data_agg("A", 1500, 80)
    t.add_variant_data_agg("B", 1200, 80)
    t.add_variant_data_agg("C", 1000, 50)
    res = t.evaluate(seed=52, return_samples=False)
    assert [r["posterior_mean"] for r in res] == [0.05363, 0.06703, 0.05045]
    assert res[1]["prob_being_best"] == pytest.approx(0.89, abs=0.02)


def test_binary_pbb_matches_exact_integral():
    t = BinaryBayesianTest()
    t.add_variant_data_agg("A", 1000, 100)
    t.add_variant_data_agg("B", 1000, 120)
    pbb = t.probabs_of_being_best(sim_count=200_000, seed=1)["B"]
    exact = prob_b_beats_a(100.5, 900.5, 120.5, 880.5)
    assert pbb == pytest.approx(exact, abs=0.005)


def test_binary_raw_equals_agg_and_append():
    t = BinaryBayesianTest()
    t.add_variant_data("A", [0, 1, 1, 0, 0])
    with pytest.warns(UserWarning):
        t.add_variant_data_agg("A", 5, 2, replace=False)
    assert t.data["A"]["totals"] == 10 and t.data["A"]["positives"] == 4


def test_binary_credible_interval_matches_exact_beta_quantiles():
    rng = np.random.default_rng(0)
    for i, x in enumerate(rng.binomial(800, 0.08, 20)):
        t = BinaryBayesianTest()
        t.add_variant_data_agg("A", 800, int(x))
        lo, hi = t.evaluate(sim_count=200_000, seed=i, return_samples=False)[0][
            "credible_interval"
        ]
        exact = stats.beta.ppf([0.025, 0.975], x + 0.5, 800 - x + 0.5)
        assert (lo, hi) == pytest.approx(tuple(exact), abs=1e-3)


def test_relative_difference_and_reference():
    t = BinaryBayesianTest()
    t.add_variant_data_agg("A", 10_000, 1000)
    t.add_variant_data_agg("B", 10_000, 1100)
    r = t.evaluate(seed=0)
    assert r[0]["relative_difference"] == 0.0
    assert r[1]["relative_difference"] == pytest.approx(0.10, abs=0.01)
    r_b = t.evaluate(seed=0, reference="B")
    assert r_b[0]["relative_difference"] == pytest.approx(-0.09, abs=0.01)
    assert len(r[1]["posterior_samples"]) == 20000


# --------------------------------------------------------------------- normal
def test_normal_close_to_t_interval_with_vague_prior():
    rng = np.random.default_rng(3)
    x = rng.normal(50, 10, 400)
    t = NormalBayesianTest(NormalInverseGamma(w_prior=0.0001))
    t.add_variant_data("A", x)
    t.add_variant_data("B", x + 1)
    r = t.evaluate(sim_count=200_000, seed=1)[0]
    se = x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), se)
    assert r["posterior_mean"] == pytest.approx(x.mean(), abs=1e-3)
    assert r["credible_interval"][0] == pytest.approx(lo, abs=0.1)
    assert r["credible_interval"][1] == pytest.approx(hi, abs=0.1)


# ------------------------------------------------------- other conjugate tests
def test_exponential_posterior_mean_closed_form():
    rng = np.random.default_rng(2)
    t = ExponentialBayesianTest()
    t.add_variant_data("A", rng.exponential(30, 3000))
    t.add_variant_data("B", rng.exponential(25, 3000))
    r = t.evaluate(seed=1, min_is_best=True)
    for row in r:
        assert row["posterior_mean"] == pytest.approx(
            np.mean(row["posterior_samples"]), rel=0.01
        )
    assert r[1]["prob_being_best"] > 0.99


def test_delta_lognormal_mean_and_min_positive():
    rng = np.random.default_rng(4)
    n = 20_000
    x = np.where(rng.random(n) < 0.3, rng.lognormal(3, 1, n), 0)
    t = DeltaLognormalBayesianTest()
    t.add_variant_data("A", x)
    t.add_variant_data("B", x)
    r = t.evaluate(seed=1)[0]
    true_mean = 0.3 * np.exp(3 + 0.5)
    assert r["posterior_mean"] == pytest.approx(true_mean, rel=0.05)
    dust = np.concatenate([x, np.full(500, 1e-8)])
    t = DeltaLognormalBayesianTest(DeltaLognormal(min_positive=1e-6))
    t.add_variant_data("A", dust)
    assert t.data["A"]["positives"] == int((x > 0).sum())


def test_delta_normal_poisson_discrete_run():
    rng = np.random.default_rng(5)
    dn = DeltaNormalBayesianTest()
    dn.add_variant_data(
        "A", np.where(rng.random(3000) < 0.2, rng.normal(5, 20, 3000), 0)
    )
    dn.add_variant_data(
        "B", np.where(rng.random(3000) < 0.2, rng.normal(9, 20, 3000), 0)
    )
    assert dn.evaluate(seed=1)[1]["prob_being_best"] > 0.8

    po = PoissonBayesianTest()
    po.add_variant_data("A", rng.poisson(2.0, 2000))
    po.add_variant_data("B", rng.poisson(2.3, 2000))
    r = po.evaluate(seed=1)
    assert r[1]["posterior_mean"] == pytest.approx(2.3, abs=0.1)

    ds = DiscreteBayesianTest(DirichletMultinomial([1, 2, 3, 4, 5]))
    ds.add_variant_data("A", [1, 2, 3, 3, 4])
    ds.add_variant_data_agg("A", [0, 0, 0, 0, 5], replace=False)
    assert ds.data["A"]["concentration"].tolist() == [1, 1, 2, 1, 5]


# --------------------------------------------------------------------- priors
def test_binary_fit_priors_and_prior_fraction():
    history = np.random.default_rng(0).binomial(1, 0.3, 5000)
    positives = history.sum()
    test = BinaryBayesianTest().fit_priors(history)
    prior = {"a_prior": positives + 0.5, "b_prior": 5000 - positives + 0.5}
    assert test.model.prior == prior

    test.add_variant_data_agg("control", 1000, 100)
    test.add_variant_data_agg("treatment", 1500, 150)
    row = test.evaluate(seed=0, prior_fraction=0.2)[0]
    # the prior keeps its mean and is worth 20% of the average variant (1250)
    prior_mean = prior["a_prior"] / (prior["a_prior"] + prior["b_prior"])
    assert row["posterior_mean"] == pytest.approx(
        (250 * prior_mean + 100) / (250 + 1000), abs=1e-5
    )
    assert test.model.prior == prior  # weighting does not change the stored prior


def _draws(rng, n, scale):
    """Data of each likelihood; ``scale`` 1 is the history, 2 the experiment."""
    zero_inflated = rng.random(n) < 0.3
    return {
        "poisson": rng.poisson(2.5 * scale, n),
        "exponential": rng.exponential(30 * scale, n),
        "normal": rng.normal(10 * scale, 3, n),
        "delta_normal": np.where(zero_inflated, rng.normal(50 * scale, 10, n), 0),
        "delta_lognormal": np.where(
            zero_inflated, rng.lognormal(3 + scale - 1, 0.8, n), 0
        ),
        "discrete": rng.integers(1 + 2 * (scale - 1), 6, n),
    }


EXPERIMENTS = {
    "poisson": PoissonBayesianTest,
    "exponential": ExponentialBayesianTest,
    "normal": NormalBayesianTest,
    "delta_normal": DeltaNormalBayesianTest,
    "delta_lognormal": DeltaLognormalBayesianTest,
    "discrete": lambda: DiscreteBayesianTest(DirichletMultinomial([1, 2, 3, 4, 5])),
}


@pytest.mark.parametrize("likelihood", list(EXPERIMENTS))
def test_fit_priors_dominate_data_when_weighted_heavily(likelihood):
    rng = np.random.default_rng(0)
    history = _draws(rng, 5000, scale=1)[likelihood]
    data = _draws(rng, 1000, scale=2)[likelihood]
    test = EXPERIMENTS[likelihood]().fit_priors(history)
    test.add_variant_data("control", data)
    test.add_variant_data("treatment", data)

    # the posterior follows the data with a negligible prior, the history with a huge one
    def posterior_mean(prior_fraction):
        row = test.evaluate(seed=0, prior_fraction=prior_fraction, return_samples=False)
        return row[0]["posterior_mean"]

    assert posterior_mean(1e-6) == pytest.approx(np.mean(data), rel=0.15)
    assert posterior_mean(1e6) == pytest.approx(np.mean(history), rel=0.1)
