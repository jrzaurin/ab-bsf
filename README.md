# ab_bsf

Bayesian and frequentist A/B testing in one package: plan an experiment, then
analyse it with whichever school you prefer. Bayesian and frequentist, and best
friends.

| | Bayesian | Frequentist |
| --- | --- | --- |
| **Design** | — | Sample size, power, minimum detectable effect and test duration for binary, continuous and zero-inflated lognormal metrics |
| **Analysis** | Conjugate models for conversions, normal, Poisson, exponential, delta-normal, delta-lognormal and categorical metrics: posterior means, credible intervals, probability of being best, expected loss, relative differences | Welch t-tests with CUPED and winsorization, a two-part test for zero-inflated metrics, difference-in-differences, fixed and mixed effects regressions, multiple-testing correction |

## Installation

Not on PyPI yet. From a clone of the repository (Python 3.10+):

```bash
pip install .            # or: uv sync --extra dev
```

## Quick tour

### Bayesian

```python
from ab_bsf.bayesian.experiments import BinaryBayesianTest

test = BinaryBayesianTest()
test.add_variant_data_agg("control", totals=1000, positives=110)
test.add_variant_data_agg("treatment", totals=1000, positives=150)

test.probabs_of_being_best(sim_count=20000, seed=1)   # {'control': 0.0038, 'treatment': 0.9962}
test.evaluate(seed=1)    # one dict per variant: posterior_mean, credible_interval,
                         # relative_difference, prob_being_best, expected_loss, ...
```

Each experiment class pairs a likelihood with its conjugate model (in
`ab_bsf.bayesian.models`), shared by all variants: `NormalBayesianTest`,
`PoissonBayesianTest`, `ExponentialBayesianTest`, `DeltaNormalBayesianTest`,
`DeltaLognormalBayesianTest` and `DiscreteBayesianTest`. The prior is
uninformative unless you pass a model with your own, e.g.
`BinaryBayesianTest(BetaBernoulli(a_prior=2, b_prior=8))`.

### Priors from historical data

To use historical data as the prior, fit it with `fit_priors`. At analysis time,
`prior_fraction` sets how much it weighs: `0.2` makes the prior worth 20% of the
observations of the average variant.

```python
from ab_bsf.bayesian.experiments import DeltaLognormalBayesianTest

test = DeltaLognormalBayesianTest().fit_priors(historical_revenue)
test.add_variant_data("control", control_revenue)
test.add_variant_data("treatment", treatment_revenue)
test.evaluate(seed=1, prior_fraction=0.2)
```

The fitted prior is in `test.model.prior`, and `evaluate` leaves it unchanged, so
you can re-evaluate with other fractions.

### Frequentist design

```python
from ab_bsf.frequentist.experiments import BinaryFrequentistTest

design = BinaryFrequentistTest(n1=1000, p1=0.05, p1_observed_var=0.0475)
design.sample_size_given_power(p2=0.10)                  # (235, 235)
design.minimum_detectable_effect()                       # (relative MDE, p2_min) at n1
design.test_duration(daily_observations=100, mde=0.3)    # dict, "duration_days": 17.0
design.mde_given_duration(daily_observations=100, days=14)   # dict, "mde_relative": 0.41
```

`ContinuousFrequentistTest` and `DeltaLognormalFrequentistTest` expose the same
methods. Pass `n_variants` for a Bonferroni-corrected multi-arm design.

### Frequentist analysis

```python
from ab_bsf.datasets import make_conversion_data, make_did_data
from ab_bsf.frequentist.metrics import t_test, t_test_did

df = make_conversion_data()                              # synthetic data
t_test(df, "conversion")["treatment_vs_control"]
# {'mean_differences': 0.0085, 'relative_difference': 0.0697, 'p_values': 0.0025, ...}

t_test_did(make_did_data(), "post", "pre")               # per-user pre/post changes
```

Every frequentist analysis function returns
`{"<group>_vs_<reference>": {metric: value}}`. The reference group defaults to
`"control"`, or the first group alphabetically.

## Development

```bash
uv sync --extra dev
uv run pytest            # unit tests and doctests
uv run ./code_style.sh   # isort, black, flake8, mypy
```

## Acknowledgements

The Bayesian side is inspired by
[bayesian-testing](https://github.com/Matt52/bayesian-testing) (MIT, © Matus
Baniar), however, ab_bsf has its own structure: one conjugate model per
likelihood, a prior shared by all variants, and priors fitted from historical
data.

## License

MIT
