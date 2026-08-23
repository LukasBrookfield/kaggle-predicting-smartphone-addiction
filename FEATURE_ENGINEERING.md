# Feature Engineering — Smartphone Addiction

Everything here was measured on this dataset, not assumed. Numbers come from `quick_cv`
(XGBoost, 200k rows, 2 folds). Baseline to beat: **0.96028**.

**Set expectations first.** Feature engineering on this dataset is worth roughly **+0.001 AUC**.
Measured:

| Experiment | OOF AUC | Delta |
|---|---|---|
| A: raw features only | 0.96028 | baseline |
| F: raw + exact combos (NaN-propagating) | 0.96072 | +0.00044 |
| D: raw + 9 iteratively-imputed columns | 0.96100 | +0.00073 |
| **E: D + latent score & combos on imputed values** | **0.96105** | **+0.00077** |
| G: everything at once (E + F + n_missing) | 0.96087 | +0.00059 |

Note **G < E: adding every feature made things worse.** That is the most important practical
lesson here, and the reason §3 insists on one group at a time. Fold std is ~0.00035, so +0.00077
is about 2 sigma — real, but modest. Training length and ensembling are the bigger levers.

---

## 1. What actually predicts the target

Before engineering anything, understand the mechanism. Three findings from the data:

**(a) It's not screen time, it's *discretionary* screen time.**

`work_study_hours` and `gaming_hours` have *positive* raw correlation with the target
(0.251, 0.205) but *negative* logistic coefficients (−0.29, −0.34). They only look addictive
because they correlate with total screen time. Conditional on total screen time, they are
protective. Target rate by screen-time bin (rows) × work-hours bin (cols):

```
w_bin     0     1     2     3     4     5
d_bin
0      0.33  0.25  0.10  0.02  0.02   NaN     <- at low screen time, work hours drop
1      0.47  0.45  0.35  0.15  0.06  0.06        the rate from 0.33 to 0.02
2      0.57  0.55  0.54  0.42  0.23  0.15
3      0.77  0.76  0.78  0.76  0.64  0.48
4      0.92  0.91  0.91  0.92  0.91  0.83
5      0.98  0.99  0.99  0.99  0.99  0.98
6      1.00  1.00  1.00  1.00  1.00  1.00     <- above ~9.5h/day everything saturates
7      1.00  1.00  1.00  1.00  1.00  1.00
```

This is a **suppression effect**, and it is why raw correlation is a bad guide to feature
usefulness. Always look at a feature's effect *conditional on* the dominant feature.

**(b) The target saturates.** Above roughly 9.5 hours/day the rate is 1.00 regardless of anything
else. All the signal lives in the middle of the range; features only have room to help there.

**(c) The boundary is diagonal, which is why combinations work.** Single-feature AUC on complete
rows:

| Feature | AUC |
|---|---|
| full weighted latent score | **0.9275** |
| `daily + 2*social` | **0.9148** |
| `daily + social` | 0.9119 |
| `daily + weekend` | 0.9011 |
| `daily - work - gaming` (leisure) | 0.8964 |
| raw `daily_screen_time_hours` | 0.8889 |

---

## 2. The core principle: combinations, not transforms

Trees are **invariant to any monotone transform of a single feature**. `log(x)`, `sqrt(x)`,
`x**2`, rank-transform, `StandardScaler` — all produce the *identical* tree, because a tree only
ever asks "is x < threshold?" and monotone transforms preserve ordering. Every `** 0.5`
experiment in `eda_addiction_prediction.ipynb` was guaranteed to do nothing for a GBDT.

What trees genuinely struggle with is a **diagonal decision boundary**. To approximate
`daily + 2*social > 12` with axis-aligned cuts, a tree needs a staircase of many splits, each
costing depth and each fit on less data. Hand it the combination directly and one split does it.

> **The rule: for tree models, engineer combinations of features, never transforms of one feature.**
> (Transforms *do* matter for the neural net and logistic models later — different story.)

---

## 3. How to test a feature — the workflow

```python
from src.cv import load_data, prepare_categoricals, quick_cv
from src.features import make_feature_pipeline
import xgboost as xgb

X, y, X_test, ids = load_data()
X, X_test = prepare_categoricals(X, X_test, flavour="lgbm")

def mk():
    return xgb.XGBClassifier(n_estimators=20000, learning_rate=0.05, max_depth=6,
                             enable_categorical=True, tree_method="hist",
                             n_jobs=-1, random_state=42)

base = quick_cv(mk, X, y, name="base").oof_auc

for groups in [["imputed"], ["imputed", "latent"], ["combos"], ["ratios"]]:
    r = quick_cv(mk, X, y, name="+".join(groups),
                 preprocessor=make_feature_pipeline(groups))
    print(f"{'+'.join(groups):28s} {r.oof_auc:.5f}  delta {r.oof_auc - base:+.5f}")
```

Rules:

- **One group at a time.** Experiment G proves piling everything on is actively worse.
- **Threshold: +0.0005.** Fold std is ~0.00035; anything smaller is noise. Reject it.
- **Screen with XGBoost** (~40s), confirm survivors with CatBoost (~6 min).
- Log every result in `results.md`, including failures, so you don't retry them.

---

## 4. The feature list

### Tier 1 — Imputed columns ✅ MEASURED +0.00073 (best single group)

Fit an `IterativeImputer` and append the results as **9 extra columns**, keeping the raw
NaN-bearing originals. The model then sees both the honest "unknown" and a best-guess
reconstruction, and learns which to trust.

Why it works: the features are strongly collinear (`weekend`↔`daily` r=0.80), so a missing value
is genuinely recoverable. Masking known values and predicting them back gives r=0.867 for
`daily_screen_time_hours`, r=0.780 for `weekend_screen_time`, r=0.667 for `social_media_hours`.

**Must be fitted inside the fold** — `IterativeImputer` learns from data, so fitting it on the
full training set leaks validation information into the model. `run_cv(preprocessor=...)` handles
this; that parameter exists for exactly this purpose.

### Tier 2 — Latent score computed on imputed values ✅ MEASURED +0.00077 total

The best single feature on this dataset. It took the **top gain importance by a wide margin
(0.47; next feature 0.18)**.

```python
W = {"daily_screen_time_hours": 1.00, "social_media_hours":  2.33,
     "weekend_screen_time":     0.96, "gaming_hours":       -0.81,
     "work_study_hours":       -0.71, "sleep_hours":         0.21}
latent = sum(w * Xi[c] for c, w in W.items())
```

The weights come from a logistic regression fitted on complete rows. Its linear predictor scores
AUC 0.9275 alone, vs 0.8889 for the best raw feature — a lot of structure in one column.

**Critical: compute it on the IMPUTED columns, not the raw ones.** The obvious NaN-tolerant
version (`skipna=True` partial sums) **fails**:

| Latent score computed on | AUC (incomplete rows) |
|---|---|
| raw partial sums (`skipna=True`) | 0.731 |
| iteratively-imputed inputs | **0.905** |

A partial sum with a missing term is not a smaller version of the score — it is a *different
quantity on a different scale*, so rows become mutually incomparable, which is exactly what AUC
measures. Adding partial sums to the GBDT scored **−0.00020**. Impute first, then combine.

**Worth trying (untested):** replace the hardcoded weights with an **out-of-fold logistic
regression prediction** as a feature — fit LR on the training fold's imputed data, predict the
validation fold. Cleaner (no weights derived from data the model is scored on) and adapts per
fold. This is "stacking as a feature", and `features.py` implements it as the `oof_logit` group.

### Tier 3 — Exact combinations on raw values ✅ MEASURED +0.00044

NaN-propagating, so only defined on complete rows — but *exact* where defined, unlike the imputed
versions. Worth keeping alongside Tier 2; the tree picks per row.

| Feature | Formula | Why |
|---|---|---|
| `total_screen` | `daily + weekend` | total weekly exposure |
| `d_plus_2s` | `daily + 2*social` | highest-AUC 2-term combo found (0.9148) |
| `leisure` | `daily - work - gaming` | the discretionary-time insight from §1a |
| `avg_screen` | `(5*daily + 2*weekend)/7` | true weekly average |
| `other_screen` | `daily - work - gaming - social` | unclassified time |

### Tier 4 — Ratios (composition, scale-free)

Ratios answer *how* time is spent rather than how much, testing the §1a mechanism directly.

| Feature | Formula | AUC |
|---|---|---|
| `leisure_ratio` | `(daily - work - gaming) / daily` | 0.746 |
| `productive_ratio` | `(work + gaming) / daily` | 0.744 (protective) |
| `social_ratio` | `social / daily` | 0.662 |
| `portion_of_wake` | `daily / (24 - sleep)` | 0.886 |
| `social_of_leisure` | `social / (daily - work - gaming)` | untested |
| `screen_share_of_free` | `daily / (24 - sleep - work)` | untested |

Always guard the denominator (`+ 0.01`) — an `inf` will poison the split finder. Note
`portion_of_wake` scores 0.886, essentially matching raw `daily` at 0.889: it is close to a
rescaling of it, so expect little on top.

### Tier 5 — Engagement intensity

A different *axis* from screen time, so potentially non-redundant even when individually weak.

| Feature | Formula | AUC |
|---|---|---|
| `mins_per_open` | `daily * 60 / app_opens` | 0.697 |
| `notif_per_open` | `notifications / app_opens` | 0.533 |
| `opens_per_wake_hour` | `app_opens / (24 - sleep)` | untested |
| `notif_per_screen_hour` | `notifications / daily` | untested |

`notifications_per_day` (AUC 0.492) and `age` (0.502) are individually indistinguishable from
noise — yet both appeared in the top-6 gain importances, meaning the tree uses them for
*interactions*. Test dropping them; don't assume either way.

### Tier 6 — Weekend consistency (untested, cheapest remaining idea)

`weekend` and `daily` correlate at 0.80. The **residual** — how much someone's weekend deviates
from what their weekday predicts — is a "binge" signal neither raw column carries. Fit
`weekend ~ a + b*daily` by `LinearRegression` inside the fold and take the residual.

Note plain `weekend - daily` scores AUC 0.509 (useless), so if this works it will be the
*regression* residual specifically, not the difference.

### Tier 7 — Imputation residual (untested, speculative)

For rows where a value *is* present, `|raw - imputed|` measures how atypical that person is
relative to what their other features predict. Unusual rows may behave differently.

---

## 5. Things that DON'T work — verified, don't waste time

| Idea | Result | Why |
|---|---|---|
| `n_missing` count | **+0.00016** (noise) | Predicted to help; it doesn't. The EDA's MCAR finding was right. |
| Per-column `_was_missing` flags | skip | Missing rate differs by target by at most 0.0042. No signal. |
| Latent score via `skipna` partial sums | **−0.00020** | Different scale per row; breaks comparability. See Tier 2. |
| `log`/`sqrt`/`**0.5` of any single column | exactly 0 | Trees are monotone-invariant. See §2. |
| Mean/median imputation *replacing* raw columns | −0.005 | Measured in your own notebooks: 0.9628 raw vs 0.9575 imputed. |
| Bucketing `age` into bins | negative | Throws away resolution and blocks interactions. |
| One-hot encoding for CatBoost/LightGBM | negative | Both have native categorical handling that beats it. |
| Adding all feature groups at once | **worse than best group alone** | G (0.96087) < E (0.96105). |

On the categoricals generally: `gender`, `stress_level`, `academic_work_impact` shift the target
rate by only 1–2 percentage points each (`stress_level`: 71.1% / 70.6% / 71.1%). Their chi-square
p-values look significant purely because n=691k. **Statistical significance is not effect size.**
Don't build target encodings on them; there is nothing there.

---

## 6. Recommended order of work

1. Rebuild the baseline through `run_cv` and confirm ≈0.9628. **Nothing below is meaningful until
   you have a reference number from your own machine.**
2. `["imputed"]` — the measured winner, ≈ +0.0007.
3. `["imputed", "latent"]` — a further ≈ +0.0001, but it becomes the top feature by importance,
   which matters more once you ensemble.
4. `["imputed", "latent", "combos"]` — does Tier 3 stack, or is it redundant?
5. Tiers 4–7, one at a time, keeping only what clears +0.0005.
6. Re-check the winning set with **CatBoost**, not just XGBoost. Feature value is model-dependent.

Then stop and move to tuning and ensembling. Realistically this phase gets you ~0.9628 → ~0.9640.
