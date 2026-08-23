# Getting from 0.96345 → 0.97 on Predicting Smartphone Addiction

## Context

You have a public LB score of **0.96345** (ROC-AUC), produced by `catboost_baseline.ipynb`:
default CatBoost, raw features, native categorical handling, NaNs left alone. Its 5-fold CV
was 0.96279 — very close to your LB, which is great news: **your CV is trustworthy**, so you
can iterate locally and believe the numbers.

Your more elaborate `train.ipynb` (sklearn pipeline, imputation, age buckets, ratio features,
Optuna) scored **worse** (0.9575). That is the central lesson of this plan: with gradient-boosted
trees, most "standard preprocessing" is actively harmful. This plan explains why and rebuilds
from the baseline that actually works.

I ran diagnostics on your data before writing this. The evidence driving every recommendation:

| Measurement | Result | What it means |
|---|---|---|
| LightGBM, lr=0.03, 3000 trees, holdout | AUC **0.96356**, `best_iteration = 2997` | Early stopping **never fired**. The model was *still improving* when it ran out of trees. Your models are badly under-trained. |
| CatBoost default (1000 iters) `learn_error.tsv` | logloss still falling at iter 999 | Same story. |
| AUC by number of missing values in a row | 0 miss: **0.9718**<br>1: 0.9664<br>2: 0.9574<br>3: 0.9455<br>4: **0.9177** | Missing data is where your score bleeds. ~56% of rows have ≥1 missing numeric. |
| Complete-case-only model | 0.968 | Even with perfect data the signal is noisy — 0.97 is genuinely hard, not a free win. |
| Single-feature AUC | `daily_screen_time` 0.890, `weekend_screen` 0.881, `social_media` 0.858, everything else ≤0.66 | Three features carry the signal; `age`/`notifications` are pure noise (AUC ≈ 0.50). |
| Target rate by screen-time decile | 0.265 → 0.310 → … → 0.999 → **1.000**, monotonic | The label is a smooth monotone function of a latent weighted score. Great for monotonic constraints and for linear/NN models. |
| Train vs test means & missing rates | Nearly identical | No distribution shift. Don't waste time on adversarial validation. |

**Honest expectation setting:** the biggest single-subgroup AUC in your data is 0.9718 (rows with
zero missing values). Reaching **0.970 overall** is ambitious. Realistic outcome from this plan is
**0.966–0.969** from modelling alone; **0.970+ most likely requires the original dataset** you said
you can download, which adds genuinely new rows rather than re-squeezing the same ones.

**Your setup:** 16 cores, 30 GB RAM, no local GPU. You'll move heavy runs to Kaggle Notebooks
(free GPU, 30h/week) — so all code below is written to run in either place, with GPU flags noted.

---

## Step 0 — Build a CV harness before changing anything

*This is the most important step and the one beginners skip.* Right now every notebook measures
things differently (one uses F1, one uses 2 folds, one uses 5). You cannot tell which idea helped.

Create **`src/cv.py`** (a plain `.py` file you import into notebooks — stops you copy-pasting CV
loops between notebooks and getting them subtly different):

```python
def run_cv(make_model, X, y, X_test, n_splits=5, seed=42, cat_features=None):
    """Fit `make_model()` on each fold. Return (oof_preds, test_preds, fold_scores)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test))
    for tr, va in skf.split(X, y):
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=..., early_stopping=...)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_splits
    return oof, test_pred, roc_auc_score(y, oof)
```

Three things this gives you that you don't have now:

1. **A single number to compare every experiment against** — always `roc_auc_score(y, oof)`,
   always 5 folds, always seed 42.
2. **Out-of-fold (OOF) predictions**, saved to disk as `oof_<modelname>.npy`. These are the raw
   material for ensembling in Step 5. Without them you cannot stack.
3. **Test predictions averaged over folds**, which is already a small ensemble and beats
   refitting once on 100% of the data (what all three of your notebooks currently do).

Keep a `results.md` scoreboard: one line per experiment, `name | CV | LB | note`. When CV goes up
but LB goes down, you're overfitting CV — that log is how you notice.

**Also fix two real bugs while you're here:**

- [train.ipynb:12](train.ipynb) calls `col_transform.fit_transform(X_test)`. That **refits the
  imputers on the test set**, so test rows get filled with test means while train rows got train
  means — the model sees a different data distribution than it was trained on. It must be
  `.transform()`. Same bug in [train_xgboost_baseline.ipynb](train_xgboost_baseline.ipynb). The
  `run_cv` harness above avoids this class of bug entirely by fitting inside the fold loop.
- [train.ipynb:12](train.ipynb) builds the submission with `CatBoostClassifier()` — bare defaults.
  Your Optuna best params were printed and then thrown away.

---

## Step 1 — Stop destroying information (the simple wins)

These are cheap and should recover the gap between your two notebooks immediately.

**1a. Never impute for a GBDT.** XGBoost, LightGBM and CatBoost all have native NaN handling: at
each split the tree *learns* whether missing rows should go left or right. That's strictly more
expressive than filling in the mean, which tells the tree "this person had exactly average screen
time" — a claim that is false 100% of the time and actively misleading given screen time is your
strongest feature. Your own results prove this: raw NaN = 0.9628, mean-imputed = 0.9575.

**1b. Don't bucket `age`.** `bucket_age` throws away resolution to no benefit — trees find their
own splits. (`age` is noise anyway at AUC 0.502, but binning also blocks it from participating in
interactions.)

**1c. Use native categorical handling, not one-hot.** You already do this in the CatBoost baseline
(`cat_features=[...]`). LightGBM: `X[c] = X[c].astype('category')`. XGBoost: `enable_categorical=True`.
Fill categorical NaN with the literal string `"missing"` — that's a real category, not an absence.

**1d. Train much, much longer.** This is your single biggest untapped lever and it costs you
nothing but wall-clock. The rule for GBDTs is *lower learning rate + more trees + early stopping*:

| Model | Instead of | Use |
|---|---|---|
| CatBoost | defaults (1000 iters, lr auto) | `iterations=20000, learning_rate=0.02, early_stopping_rounds=300` |
| LightGBM | defaults (100 trees!) | `n_estimators=20000, learning_rate=0.02, early_stopping(300)` |
| XGBoost | defaults (100 trees) | `n_estimators=20000, learning_rate=0.02, early_stopping_rounds=300` |

Set the cap absurdly high and let early stopping choose. If `best_iteration` comes back equal to
your cap, the cap was the binding constraint and you're still leaving score on the table — that is
exactly what happened in my test (2997 out of 3000) and in your Optuna run (best iteration 499 out
of a 500 cap). **Always print `best_iteration` and check it isn't the ceiling.**

On Kaggle's GPU: `CatBoostClassifier(task_type="GPU")`, `XGBClassifier(device="cuda", tree_method="hist")`.

**1e. Add the original dataset.** You said one exists. Concatenate it onto your training rows:

```python
orig = pd.read_csv("data/original.csv")
orig = orig[train.columns.drop("id")]          # align columns exactly
train_aug = pd.concat([train.drop(columns="id"), orig], ignore_index=True)
```

Two rules that matter:
- **Only augment the training folds, never the validation fold.** If original rows leak into
  validation your CV becomes optimistic and stops matching the LB. Concatenate *inside* the fold
  loop, after the split.
- Add a `is_original` flag column so the model can learn that these rows are distributed slightly
  differently. Also try dropping it — test both, keep whichever OOF is higher.

**Expected after Step 1: ~0.965–0.967.**

---

## Step 2 — Feature engineering, added *on top of* raw features

Your `train.ipynb` FE was reasonable — the mistake was that it *replaced* the raw columns via
`remainder="drop"` and got imputed. Always keep the originals and append.

Two features are worth adding that you don't have, both motivated by the diagnostics:

**2a. `n_missing` — count of missing values in the row.** Your EDA correctly found missingness is
MCAR (missing-rate barely differs by target), and concluded indicators are worthless. That
conclusion is half right and the distinction is worth understanding:

- MCAR means missingness does **not** predict `y` directly — so per-column `_was_missing` flags are
  indeed near-useless. Your EDA was right about that.
- But `n_missing` tells the model **how much evidence it has about this particular row**. My
  measurement shows AUC collapsing from 0.9718 to 0.9177 as `n_missing` goes 0 → 4. A model that
  knows a row is information-poor can shrink that row's prediction toward the base rate, which is
  the *correct* thing to do and improves the cross-row ranking that AUC measures.

**2b. Model-based imputation as *extra columns* (not replacements).** `weekend_screen_time` and
`daily_screen_time_hours` correlate at r = 0.80, and both are top-3 predictors. When one is
missing, the other largely reconstructs it — but a tree can only do that reconstruction implicitly
and imperfectly. So do it explicitly:

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imp.fit_transform(X[num_cols])          # fit on TRAIN FOLD only
X = X.join(pd.DataFrame(X_imputed, columns=[c + "_imp" for c in num_cols], index=X.index))
```

The model now sees both the honest NaN *and* a best-guess reconstruction, and decides which to
trust. This targets precisely the rows where you're losing score. Fit the imputer inside the fold
loop — it learns from data, so fitting it on the full train set leaks.

**2c. Keep your existing ratios** (`social_media_ratio`, `portion_of_day_on_screen`,
`unaccounted_screen_time`, `screen_to_work_ratio`) but compute `unaccounted_screen_time` with
`skipna=True` as your EDA version did, so one missing component doesn't nuke the whole feature.

**How to evaluate FE:** add features **one at a time**, re-run `run_cv`, keep it only if OOF AUC
improves by more than fold-to-fold noise (your folds vary by ±0.0005, so demand ≥ +0.0005).
Adding ten features at once and seeing the score move tells you nothing about which one worked.

**Expected after Step 2: ~0.966–0.968.**

---

## Step 3 — Hyperparameter tuning, done properly

Your Optuna run had three flaws, all fixable:

1. **`N_FOLDS = 2`** — noisy. A 2-fold estimate has enough variance that Optuna optimises noise.
   Use 5. If that's too slow, tune on a 250k-row stratified subsample with 5 folds instead — a
   subsample with correct fold count beats full data with wrong fold count.
2. **`iterations = 500` hard cap** — the binding constraint, so every trial was under-trained and
   Optuna concluded "high learning rate is best" (best trial: lr = 0.26). That's not a real finding,
   it's the search compensating for too few trees. Set `iterations=20000` with
   `early_stopping_rounds=300` and tune `learning_rate` over `[0.01, 0.1]` instead.
3. **`N_TRIALS = 20`** — too few for a 6-dimensional space. Aim for 50–100 on Kaggle's GPU.

Ranges worth searching (CatBoost): `depth` 4–10, `l2_leaf_reg` 1–20 (log), `random_strength` 0–5,
`bagging_temperature` 0–2, `border_count` 128–254, `min_data_in_leaf` 1–100.

**A warning about tuning:** it typically buys +0.001–0.002. That's real but small — it is *not*
where 0.97 comes from. Do it once, don't obsess. Beginners over-invest here because it feels like
progress; ensembling (Step 5) pays far better.

---

## Step 4 — Build genuinely different models

Ensembling only works when models make *different mistakes*. Train these four, saving OOF each time:

1. **CatBoost** — native cats via ordered target statistics, native NaN. Your current best.
2. **LightGBM** — leaf-wise growth, different NaN routing. Very different error profile from CatBoost.
3. **XGBoost** — `enable_categorical=True`, depth-wise growth.
4. **A neural network (MLP)** — *the highest-value item in this plan after Step 1.* Your target rate
   curve is smooth and monotone, which is exactly what a NN models well and trees model as a
   staircase. GBDT + NN blends are the standard recipe in Kaggle tabular competitions precisely
   because the two families fail differently. You have `torch 2.13.0` installed.

   NN recipe for this data: median-impute + `StandardScaler` (NNs *do* need imputation, unlike
   trees), one-hot the 3 categoricals, append the `_was_missing` indicators (the NN can't route
   NaNs itself, so here the indicators genuinely earn their place), then
   `Linear(d,256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256,128) → … → Linear(128,1)`,
   `BCEWithLogitsLoss`, AdamW lr=1e-3 with cosine decay, batch 1024, ~30 epochs, early stop on
   fold AUC. Expect ~0.958–0.962 alone — *worse* than CatBoost, and that is fine. A weaker,
   uncorrelated model still lifts a blend.

**Also worth trying — monotonic constraints.** Your decile table shows the target rate rising
monotonically with `daily_screen_time_hours`. Telling the model to enforce that
(`monotone_constraints` in LightGBM/XGBoost, `monotone_constraints` in CatBoost) removes its
freedom to fit noisy non-monotone wiggles, which usually costs a little train fit and gains a
little generalisation. Apply to `daily_screen_time_hours`, `weekend_screen_time`,
`social_media_hours` (all +1). Test it; keep if OOF improves.

**Check diversity before blending:** `np.corrcoef(oof_a, oof_b)`. Two models at 0.999 correlation
will not help each other. Below ~0.98 is where blending starts paying real dividends.

---

## Step 5 — Ensemble (this is where the remaining score lives)

You currently do no ensembling at all. This is the largest gap between your notebook and a
leaderboard notebook.

**5a. Rank-average first (2 minutes, almost always helps).**

```python
from scipy.stats import rankdata
blend = np.mean([rankdata(p) / len(p) for p in [oof_cat, oof_lgb, oof_xgb, oof_nn]], axis=0)
```

Why *rank*-average rather than plain mean: ROC-AUC only cares about the **ordering** of your
predictions, not their values. Models on different probability scales (a NN outputs confident
0.99s, CatBoost is more conservative) get distorted by a plain average — the confident model
dominates. Converting to ranks puts everyone on the same scale first, so each model contributes
equally to the ordering. **Always rank-average when the metric is AUC.**

**5b. Weighted blend via hill-climbing.** Equal weights are rarely optimal. Greedily search weights
that maximise OOF AUC: start with the best single model, then repeatedly test adding a small
increment of each model, keep the move that improves OOF most, stop when nothing helps. ~20 lines,
reliably beats equal weights.

**5c. Stacking.** Train a meta-model on the OOF predictions as features:

```python
meta_X = np.column_stack([oof_cat, oof_lgb, oof_xgb, oof_nn])
meta = LogisticRegression(C=1.0).fit(meta_X, y)     # apply the same CV harness here too
```

Because OOF predictions are by construction out-of-fold, the meta-model never sees a prediction
made by a model that trained on that row — that's what makes stacking legitimate rather than
leaky. Add `n_missing` as an extra meta-feature: it lets the meta-model weight the base models
*differently for information-poor rows*, which is exactly where they disagree most. Keep the
meta-model simple (logistic regression / ridge); complex meta-models overfit the OOF badly.

**5d. Multi-seed bagging.** Re-run your best single config with seeds 42, 1337, 2024, 7, 99 and
average. Pure variance reduction, no thinking required, typically +0.0005–0.001. Do this last on
whatever config wins.

**Expected after Step 5: ~0.968–0.970.**

---

## Step 6 — Advanced techniques, in order of expected payoff

**6a. Per-stratum calibration (the non-obvious one, and well suited to your data).** Overall AUC
ranks *every* row against every other row, including comparing a 0-missing row against a 4-missing
row. If your model's scores mean different things in those two groups — e.g. it's systematically
over-confident on information-poor rows — the cross-group comparisons are wrong and overall AUC
suffers *even when the ranking within each group is perfect*. Fix: fit a small isotonic or Platt
calibrator per `n_missing` stratum on the OOF predictions, apply to test. Diagnose first by
comparing mean predicted vs actual rate within each stratum; only bother if they diverge.

**6b. Pseudo-labelling.** Predict on test, take the most confident rows (p > 0.99 or p < 0.01), add
them to training with their predicted labels, retrain. With 296k test rows this is a lot of extra
data. It can also confidently reinforce your own mistakes — always verify on OOF, use one round
only, and be prepared to discard it.

**6c. Feature-neutral sanity check.** `age` (AUC 0.502) and `notifications_per_day` (0.492) are
statistically indistinguishable from noise. Try dropping them. Sometimes removing noise features
helps; sometimes trees use them productively for interactions. Measure, don't assume.

**6d. 10-fold instead of 5-fold.** More training data per fold, less biased CV estimate. Doubles
runtime for maybe +0.0003. Only worth it on Kaggle's GPU for your final submission.

---

## Suggested file layout

```
src/cv.py                  # run_cv harness, seed constants  ← build this first
src/features.py            # add_features(df) — one function, used by every notebook
notebooks/01_baseline.ipynb    # Steps 0-1: rebuild clean baseline, verify ~0.966
notebooks/02_features.ipynb    # Step 2: one feature at a time
notebooks/03_tuning.ipynb      # Step 3: Optuna (run on Kaggle)
notebooks/04_models.ipynb      # Step 4: cat/lgb/xgb/nn, save oof_*.npy
notebooks/05_ensemble.ipynb    # Step 5: rank avg, hill climb, stack
results.md                     # experiment scoreboard
oof/                           # oof_*.npy, test_*.npy
```

Your existing notebooks stay untouched as a record. `train_new.csv` is misnamed — it's a
submission file, not a dataset; rename it to `sub_catboost_pipeline.csv` to avoid confusion later.

---

## Verification

- **After Step 0:** re-run the CatBoost baseline through `run_cv`. It must reproduce ≈0.9628.
  If it doesn't, the harness is wrong — fix it before going further.
- **After every change:** OOF AUC in `results.md`, compared against fold-noise (±0.0005). Reject
  anything that doesn't clear it.
- **Every 3–4 experiments:** actually submit to Kaggle and record the LB score next to the CV score.
  You currently have one CV/LB pair (0.96279 / 0.96345) and it correlates well. If CV starts rising
  while LB stalls, you're overfitting CV — back off, simplify, use more folds.
- **Before each submission:** check `sub.shape == (296302, 2)`, columns `id,addicted_label`,
  predictions are probabilities in [0,1] (not hard 0/1 — that destroys AUC), and mean prediction
  ≈ 0.709. Your `sub_train_xgboost_baseline.csv` has mean 0.730, a sign it was mis-calibrated.
- **Final submission:** Kaggle lets you select 2. Pick your best-CV ensemble *and* one robust
  single model — if the ensemble overfit the public LB, the simpler model protects you on private.
