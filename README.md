# ab_bsf

Bayesian and frequentist A/B testing, structured like
[bayesian-testing](https://github.com/Matt52/bayesian-testing) (`experiments` + `metrics`)
for both schools:

```
ab_bsf/
├── bayesian/
│   ├── experiments/   BinaryBayesianTest, NormalBayesianTest, DeltaLognormalBayesianTest,
│   │                  DeltaNormalBayesianTest, ExponentialBayesianTest, PoissonBayesianTest,
│   │                  DiscreteBayesianTest, BinaryDidTest (+ BaseBayesianTest, BaseDidTest)
│   └── metrics/       posteriors.py (conjugate samplers), evaluation.py (P(best), loss, CIs, relative diffs)
├── frequentist/
│   ├── experiments/   design: BinaryFrequentistTest, ContinuousFrequentistTest,
│   │                  DeltaLognormalFrequentistTest, clustered.py, functional.py
│   └── metrics/       evaluation: t_tests.py, regressions.py, preprocessing.py, multiple_testing.py
├── datasets/          synthetic generators
└── utils/             alias_params, ...
```

```bash
pip install -e ".[dev]"
pytest            # 98 tests incl. doctests
```

## Quick tour

```python
from ab_bsf.bayesian.experiments import BinaryBayesianTest
t = BinaryBayesianTest()
t.add_variant_data_agg("control", totals=1000, positives=110)
t.add_variant_data_agg("treatment", totals=1000, positives=150)
t.evaluate(seed=1)          # posterior_mean, credible_interval, relative_difference,
                            # relative_credible_interval, prob_being_best, expected_loss, posterior_samples
t.evaluate_with_did(pre_rates={"control": 0.10, "treatment": 0.12})

from ab_bsf.frequentist.experiments import BinaryFrequentistTest
d = BinaryFrequentistTest(n1=1000, p1=0.05)          # one-tailed, pooled variance by default
d.sample_size_given_power(p2=0.10)                   # (341, 341)
d.minimum_detectable_effect()                        # (relative MDE, p2_min) at n1
d.test_duration(daily_observations=500, mde=0.2)     # days beyond the n1 already collected
d.mde_given_duration(daily_observations=500, days=14)

from ab_bsf.frequentist.metrics import t_test, t_test_did, t_test_delta_lognormal
t_test(df, "revenue", covariate_col="revenue_pre", winsorize_upper=0.99)   # Welch + CUPED
t_test_did(df, "revenue", "revenue_pre")                                   # per-user changes
```

Frequentist results are `{"<group>_vs_<reference>": {metric: value}}`; the reference
defaults to `"control"` (or the first group alphabetically).

## Provenance

This package was rebuilt from conversations about the original library (Jul 2025 – Mar 2026).
The original files themselves were not available, so:

| Part | Source |
|---|---|
| `bayesian/metrics/posteriors.py`, the Beta/NIG/Gamma/Dirichlet maths | bayesian-testing (MIT, © Matus Baniar), which ab_bsf mirrored |
| Bayesian class names, `relative_difference` / `relative_credible_interval` / `posterior_samples` outputs, `add_variant_data_did` (Normal), `evaluate_with_did` (Binary), `BinaryDidTest` | recovered from chats |
| `ContinuousFrequentistTest`, `DeltaLognormalFrequentistTest`, `alias_params` usage, `n_variants`/`alpha_`, `test_duration` | recovered from chats |
| `BaseFrequentistTest`, `BinaryFrequentistTest` internals, `alias_params` body | reconstructed from how they were called |
| `t_test`, `t_test_did`, `t_test_did_log`, `t_test_did_regr`, `t_test_delta_lognormal`, `winsorize`, `cuped_adjustment`, fixed/mixed regressions | recovered signatures and behaviour; bodies rewritten |
| `datasets` | new (the `session_data.parquet` could not be recovered) |

## Deliberate changes from the chat versions

Statistics:

1. **Arcsine sample size / power / MDE**: the "1/4 factor" added in July 2025 was wrong
   (4x too small samples). Fixed; matches statsmodels.
2. **`t_test_did_regr`**: SEs are clustered by unit. Plain HC3 on the long panel treats
   each user's pre and post rows as independent and overstates the SE (by >1.5x in the tests).
3. **`t_test`** is Welch throughout (one version mixed a pooled-variance p-value with a Welch CI);
   relative CIs use the delta method for a ratio of means, not `se / |control_mean|`.
4. **`t_test_delta_lognormal`**: magnitude lift is the geometric-mean ratio `exp(diff) - 1`,
   consistent with its log-scale test.
5. **`test_duration`** never returns negative days (`already_collected=False` plans from zero).
6. **`mde_given_duration`** uses a fixed power; the chat's "MDE at achievable power" was circular.
7. **`winsorize`** defaults to `"pooled"` caps; per-group caps can shave off part of the treatment effect.
8. **DeltaLognormal design**: the lift split between conversion and magnitude is
   multiplicative (`conversion_share`), and two-tailed power includes the opposite tail.
9. **Bayesian delta-lognormal** allows negative `m_prior` (log means can be negative) and `min_positive`
   (drops dust before logging).

Left out on purpose:

* **DiD for Exponential / Delta-Lognormal via `|post - pre|` or positive-only changes**: those transforms
  drop the sign of the change or misclassify decreases, so the "DiD" they report is not a DiD.
  For per-user pre/post data use `NormalBayesianTest.add_variant_data_did` or `t_test_did(_log)`.
* **"Binary DiD" via `post == 1`**: that is just the post-period rate. Use `evaluate_with_did`
  (known pre rates) or `BinaryDidTest` (pre-period counts).
* PyMC (MCMC) exponential test, linearmodels two-way panel FE: optional heavy dependencies, not rebuilt.
