"""
Cross-validation harness for the "Predicting Smartphone Addiction" competition.

WHY THIS FILE EXISTS
--------------------
Every experiment has to be measured the same way, or you cannot tell which idea
actually helped. `run_cv` is the single source of truth: same folds, same seed,
same metric, every single time.

It also produces the two things you need for ensembling later:

  * out-of-fold (OOF) predictions  -> the raw material for blending/stacking
  * fold-averaged test predictions -> a better submission than refitting once
                                      on 100% of the data

A note on speed. Running 5 folds on all 691k rows takes real time on CPU, so
this module gives you two dials:

  * `sample=`      train/validate on a stratified subsample of the rows
  * `max_folds=`   stop after the first N folds

`quick_cv()` presets both for fast iteration. Use it to *screen* ideas, then
confirm the winners with a full `run_cv()` before you trust the number.

Measured on this machine (16 cores, no GPU), quick_cv at 200k rows / 2 folds:

    XGBoost    31s      <-- use these two for screening features
    LightGBM   71s
    CatBoost  358s      <-- ~5x slower; save it for confirmation runs

So: iterate on features with XGBoost, then confirm the survivors with CatBoost
(which scores highest on this dataset) before submitting.

One trap when reading `result.importances`: LightGBM defaults to counting how
many times a feature was *split on*, which is not the same as how much it
*helped* -- it ranks the pure-noise `notifications_per_day` first. Pass
`importance_type="gain"` to LGBMClassifier so the numbers mean something.
XGBoost and CatBoost already report gain-style importances.

TYPICAL USE
-----------
    from src.cv import load_data, prepare_categoricals, quick_cv, run_cv
    from catboost import CatBoostClassifier

    X, y, X_test, test_ids = load_data()
    X, X_test = prepare_categoricals(X, X_test, flavour="catboost")

    def make_model():
        return CatBoostClassifier(
            iterations=20000, learning_rate=0.02, eval_metric="AUC",
            random_seed=42, verbose=False, allow_writing_files=False,
        )

    res = quick_cv(make_model, X, y, cat_features=CAT_COLS, name="cat_v1")
    print(res.summary())          # screen the idea in ~1-2 min

    res = run_cv(make_model, X, y, X_test, cat_features=CAT_COLS, name="cat_v1")
    res.save()                    # confirm it, then keep the OOF for blending
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Project-wide constants. Change these in ONE place, never in a notebook.
# ---------------------------------------------------------------------------

SEED = 42
N_SPLITS = 5
DATA_DIR = "data"
OOF_DIR = "oof"

TARGET = "addicted_label"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str = DATA_DIR):
    """Load train/test. Returns (X, y, X_test, test_ids).

    NaNs are deliberately left untouched: XGBoost, LightGBM and CatBoost all
    learn a split direction for missing values, which is strictly better than
    imputing a value that is wrong for every single row.
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    y = train.pop(TARGET)
    X = train.drop(columns="id")

    test_ids = test["id"]
    X_test = test.drop(columns="id")

    return X, y, X_test[X.columns], test_ids


def prepare_categoricals(X, X_test=None, cat_cols=CAT_COLS, flavour="catboost"):
    """Encode the 3 categorical columns the way `flavour` expects.

    catboost -> fill NaN with the literal string "missing" (a real category,
                not an absence) and cast to str.
    lgbm/xgb -> pandas 'category' dtype, which both libraries handle natively
                when told to (LightGBM automatically, XGBoost via
                enable_categorical=True). NaN stays NaN.
    """
    X = X.copy()
    X_test = None if X_test is None else X_test.copy()

    for c in cat_cols:
        if flavour == "catboost":
            X[c] = X[c].fillna("missing").astype(str)
            if X_test is not None:
                X_test[c] = X_test[c].fillna("missing").astype(str)
        else:
            # Build the category list from train+test so the codes line up.
            levels = pd.Index(sorted(set(X[c].dropna().unique())))
            X[c] = pd.Categorical(X[c], categories=levels)
            if X_test is not None:
                X_test[c] = pd.Categorical(X_test[c], categories=levels)

    return (X, X_test) if X_test is not None else X


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Everything one cross-validation run produced."""

    name: str
    oof: np.ndarray  # OOF prediction per training row (NaN where not scored)
    test_pred: np.ndarray | None  # fold-averaged test predictions
    fold_scores: list[float]
    oof_auc: float
    best_iterations: list[int | None]
    elapsed: float
    n_rows_used: int
    importances: pd.Series | None = None
    fitted_rows: np.ndarray = field(default=None, repr=False)

    def summary(self) -> str:
        folds = "  ".join(f"{s:.5f}" for s in self.fold_scores)
        out = [
            f"{self.name}",
            f"  fold AUCs   : {folds}",
            f"  mean +/- std: {np.mean(self.fold_scores):.5f} +/- {np.std(self.fold_scores):.5f}",
            f"  OOF AUC     : {self.oof_auc:.5f}   <-- compare experiments on THIS number",
            f"  rows used   : {self.n_rows_used:,}",
            f"  time        : {self.elapsed:.1f}s",
        ]
        iters = [i for i in self.best_iterations if i is not None]
        if iters:
            out.append(f"  best_iters  : {iters}")
        return "\n".join(out)

    def save(self, out_dir: str = OOF_DIR) -> None:
        """Persist OOF + test predictions so Step 5 (ensembling) can load them."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/oof_{self.name}.npy", self.oof)
        if self.test_pred is not None:
            np.save(f"{out_dir}/test_{self.name}.npy", self.test_pred)
        print(f"saved -> {out_dir}/oof_{self.name}.npy")


# ---------------------------------------------------------------------------
# Library-specific plumbing
#
# The three boosting libraries all support early stopping, but each spells it
# differently. run_cv hides that so your notebooks stay clean.
# ---------------------------------------------------------------------------


def _flavour(model) -> str:
    mod = type(model).__module__
    if mod.startswith("lightgbm"):
        return "lgbm"
    if mod.startswith("xgboost"):
        return "xgb"
    if mod.startswith("catboost"):
        return "catboost"
    return "sklearn"


def _fit_fold(model, X_tr, y_tr, X_va, y_va, cat_features, early_stopping_rounds):
    """Fit one fold with early stopping wired up correctly for this library."""
    kind = _flavour(model)

    if kind == "catboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_features,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=early_stopping_rounds is not None,
            verbose=False,
        )

    elif kind == "lgbm":
        import lightgbm as lgb

        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            eval_metric="auc",
            callbacks=callbacks,
        )

    elif kind == "xgb":
        # XGBoost takes early_stopping_rounds on the constructor, not on fit().
        if early_stopping_rounds and model.get_params().get("early_stopping_rounds") is None:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
        if model.get_params().get("eval_metric") is None:
            model.set_params(eval_metric="auc")
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    else:  # plain sklearn estimator: no early stopping concept
        model.fit(X_tr, y_tr)

    return model


def _best_iteration(model):
    for attr in ("best_iteration_", "best_iteration"):
        v = getattr(model, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    if hasattr(model, "get_best_iteration"):
        try:
            return int(model.get_best_iteration())
        except Exception:
            return None
    return None


def _importances(model, columns):
    imp = getattr(model, "feature_importances_", None)
    if imp is None or len(imp) != len(columns):
        return None
    return pd.Series(imp, index=columns)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def run_cv(
    make_model,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame | None = None,
    *,
    name: str = "model",
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    cat_features: list[str] | None = None,
    early_stopping_rounds: int | None = 200,
    sample: int | None = None,
    max_folds: int | None = None,
    preprocessor=None,
    verbose: bool = True,
) -> CVResult:
    """Stratified K-fold cross-validation returning OOF + test predictions.

    Parameters
    ----------
    make_model : callable
        Zero-argument factory returning a FRESH, unfitted model. It must be a
        factory, not a model -- reusing one instance across folds would let
        fold 2 continue training from fold 1 and quietly leak.
    X, y : training features / target. Leave NaNs in place for GBDTs.
    X_test : optional. If given, test predictions are averaged over the folds.
    cat_features : column names to pass to CatBoost's `cat_features`. Ignored
        by the other libraries (they read the pandas 'category' dtype instead).
    early_stopping_rounds : stop a fold when validation AUC has not improved
        for this many rounds. Set your model's n_estimators/iterations absurdly
        high and let this pick the real number. Pass None to disable.
    sample : train on a stratified subsample of this many rows. The fastest way
        to iterate. Absolute AUC will be lower than the full run, but the
        *ranking* of ideas is usually preserved.
    max_folds : only run the first N folds. `max_folds=2` roughly halves the
        time of a 5-fold run while keeping each fold's 80% training size.
    preprocessor : optional callable returning an object with
        fit_transform/transform (e.g. an IterativeImputer or a Pipeline). It is
        fitted on the TRAINING FOLD ONLY and applied to validation and test.
        This is how you use a learned transform without leaking.

    Returns
    -------
    CVResult
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if sample is not None and sample < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=sample, random_state=seed, stratify=y
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_folds_to_run = min(max_folds or n_splits, n_splits)

    oof = np.full(len(X), np.nan)
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    fold_scores: list[float] = []
    best_iters: list[int | None] = []
    imps: list[pd.Series] = []
    scored_rows: list[np.ndarray] = []

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        if fold >= n_folds_to_run:
            break

        t_fold = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_te = X_test

        # Learned preprocessing: fit on the training fold only, never on the
        # validation fold and never on the full dataset.
        if preprocessor is not None:
            prep = preprocessor()
            X_tr = prep.fit_transform(X_tr, y_tr)
            X_va = prep.transform(X_va)
            if X_te is not None:
                X_te = prep.transform(X_te)

        model = make_model()
        model = _fit_fold(
            model, X_tr, y_tr, X_va, y_va, cat_features, early_stopping_rounds
        )

        p_va = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = p_va
        scored_rows.append(va_idx)
        fold_scores.append(roc_auc_score(y_va, p_va))
        best_iters.append(_best_iteration(model))

        imp = _importances(model, list(X_tr.columns))
        if imp is not None:
            imps.append(imp)

        if X_te is not None:
            test_pred += model.predict_proba(X_te)[:, 1] / n_folds_to_run

        if verbose:
            bi = best_iters[-1]
            bi_txt = f"  best_iter={bi}" if bi is not None else ""
            print(
                f"  fold {fold + 1}/{n_folds_to_run}  AUC={fold_scores[-1]:.5f}"
                f"  ({time.time() - t_fold:.0f}s){bi_txt}"
            )

    elapsed = time.time() - t0
    scored = np.concatenate(scored_rows)
    oof_auc = roc_auc_score(y.iloc[scored], oof[scored])

    result = CVResult(
        name=name,
        oof=oof,
        test_pred=test_pred,
        fold_scores=fold_scores,
        oof_auc=oof_auc,
        best_iterations=best_iters,
        elapsed=elapsed,
        n_rows_used=len(X),
        importances=(pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
                     if imps else None),
        fitted_rows=scored,
    )

    if verbose:
        print(result.summary())
        _warn_if_capped(result, make_model)

    return result


def _warn_if_capped(result: CVResult, make_model) -> None:
    """Tell the user when the tree budget, not the data, was the limiting factor.

    If best_iteration lands on the cap, the model was still improving when it
    ran out of trees -- raise n_estimators/iterations and you get free score.
    """
    iters = [i for i in result.best_iterations if i is not None]
    if not iters:
        return
    try:
        p = make_model().get_params()
    except Exception:
        return
    cap = p.get("n_estimators") or p.get("iterations")
    if cap and max(iters) >= cap - 1:
        print(
            f"  WARNING: best_iteration hit the cap of {cap}. The model was still "
            f"improving -- raise n_estimators/iterations and re-run."
        )


def quick_cv(make_model, X, y, X_test=None, *, sample=200_000, max_folds=2, **kwargs):
    """Fast screening run: 200k rows, first 2 folds. For iterating on ideas.

    The absolute AUC is lower than a full run (less training data), so never
    compare a quick_cv number against a run_cv number. Compare quick against
    quick, then confirm the winner with the full run_cv.
    """
    return run_cv(make_model, X, y, X_test, sample=sample, max_folds=max_folds, **kwargs)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def make_submission(test_pred, test_ids, path: str) -> pd.DataFrame:
    """Write a submission and sanity-check it before you burn a daily attempt."""
    sub = pd.DataFrame({"id": np.asarray(test_ids), TARGET: np.asarray(test_pred)})

    assert sub.shape == (296302, 2), f"expected 296302 rows, got {sub.shape[0]}"
    assert sub[TARGET].between(0, 1).all(), "predictions must be probabilities in [0, 1]"
    assert sub[TARGET].nunique() > 2, "these look like hard 0/1 labels -- that destroys AUC"

    mean = sub[TARGET].mean()
    print(f"{path}: {len(sub):,} rows, mean prediction {mean:.4f} (train base rate 0.7094)")
    if abs(mean - 0.7094) > 0.03:
        print("  WARNING: mean prediction is far from the base rate -- likely miscalibrated.")

    sub.to_csv(path, index=False)
    return sub
