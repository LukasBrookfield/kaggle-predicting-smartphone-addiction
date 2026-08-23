"""
Feature engineering for the smartphone-addiction competition.

See FEATURE_ENGINEERING.md for the reasoning and the measured deltas behind every
group here. The short version:

  * Trees are invariant to monotone transforms of a single column, so `log(x)`
    buys nothing. Engineer *combinations*, which cut the diagonal boundaries
    that axis-aligned splits approximate badly.
  * Imputing the numeric columns and appending them as EXTRA columns was the
    best measured group (+0.00073).
  * The weighted "latent score" must be computed on the IMPUTED columns. The
    NaN-tolerant `skipna=True` version scores 0.731 vs 0.905 and hurts the model.

USAGE
-----
Groups are toggleable so you can A/B test one at a time -- adding everything at
once measured WORSE than the best single group.

    from src.cv import load_data, prepare_categoricals, quick_cv
    from src.features import make_feature_pipeline

    X, y, X_test, ids = load_data()
    X, X_test = prepare_categoricals(X, X_test, flavour="lgbm")

    r = quick_cv(mk_model, X, y, name="imp",
                 preprocessor=make_feature_pipeline(["imputed", "latent"]))

`make_feature_pipeline` returns an object with fit_transform/transform, which is
the contract `run_cv(preprocessor=...)` expects. Everything learned from data
(the imputer, the weekend regression, the OOF logistic) lives inside it, so it is
fitted on the training fold only and never leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from .cv import NUM_COLS, SEED

# Column shorthands, so the formulas below read like the document.
DAILY = "daily_screen_time_hours"
SOCIAL = "social_media_hours"
GAMING = "gaming_hours"
WORK = "work_study_hours"
SLEEP = "sleep_hours"
NOTIF = "notifications_per_day"
OPENS = "app_opens_per_day"
WEEKEND = "weekend_screen_time"

# Weights of the latent "addiction score", read off a logistic regression fitted
# on complete rows and rescaled so DAILY has weight 1.0. The linear predictor
# scores AUC 0.9275 on its own vs 0.8889 for the best raw feature.
LATENT_W = {
    DAILY: 1.00,
    SOCIAL: 2.33,
    WEEKEND: 0.96,
    GAMING: -0.81,
    WORK: -0.71,
    SLEEP: 0.21,
}

GROUPS = [
    "imputed",       # +0.00073  9 iteratively-imputed columns, appended
    "latent",        # +0.00077  weighted score + combos, on imputed values
    "combos",        # +0.00044  exact combinations on raw values (NaN-propagating)
    "ratios",        #           composition: how the time is spent
    "intensity",     #           engagement per app-open / per waking hour
    "weekend_resid", #           untested: weekend deviation from its weekday prediction
    "imp_resid",     #           untested: |raw - imputed|, how atypical the row is
    "oof_logit",     #           untested: OOF logistic prediction as a feature
    "n_missing",     # +0.00016  NOISE -- kept only so you can reproduce the null result
]

EPS = 0.01  # denominator guard; an inf would poison the split finder


# ---------------------------------------------------------------------------
# Individual groups. Each takes the raw frame X and (where needed) the imputed
# frame Xi, and returns a DataFrame of NEW columns only.
# ---------------------------------------------------------------------------


def _combos(d: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Exact combinations. Works on either raw or imputed values."""
    s = suffix
    return pd.DataFrame(
        {
            f"total_screen{s}": d[DAILY] + d[WEEKEND],
            f"d_plus_2s{s}": d[DAILY] + 2 * d[SOCIAL],
            f"leisure{s}": d[DAILY] - d[WORK] - d[GAMING],
            f"avg_screen{s}": (5 * d[DAILY] + 2 * d[WEEKEND]) / 7,
            f"other_screen{s}": d[DAILY] - d[WORK] - d[GAMING] - d[SOCIAL],
        },
        index=d.index,
    )


def _ratios(d: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    s = suffix
    leisure = d[DAILY] - d[WORK] - d[GAMING]
    return pd.DataFrame(
        {
            f"social_ratio{s}": d[SOCIAL] / (d[DAILY] + EPS),
            f"productive_ratio{s}": (d[WORK] + d[GAMING]) / (d[DAILY] + EPS),
            f"leisure_ratio{s}": leisure / (d[DAILY] + EPS),
            f"social_of_leisure{s}": d[SOCIAL] / (leisure.clip(lower=0) + EPS),
            f"portion_of_wake{s}": d[DAILY] / (24 - d[SLEEP]),
            f"screen_share_of_free{s}": d[DAILY] / ((24 - d[SLEEP] - d[WORK]).clip(lower=EPS)),
        },
        index=d.index,
    )


def _intensity(d: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    s = suffix
    opens = d[OPENS].replace(0, np.nan)
    return pd.DataFrame(
        {
            f"mins_per_open{s}": d[DAILY] * 60 / opens,
            f"notif_per_open{s}": d[NOTIF] / opens,
            f"opens_per_wake_hour{s}": d[OPENS] / (24 - d[SLEEP]),
            f"notif_per_screen_hour{s}": d[NOTIF] / (d[DAILY] + EPS),
        },
        index=d.index,
    )


def _latent(d: pd.DataFrame, suffix: str = "") -> pd.Series:
    """Weighted latent addiction score.

    Compute this on IMPUTED values. The `skipna=True` partial-sum version scores
    0.731 vs 0.905 on incomplete rows, because a partial sum is a different
    quantity on a different scale rather than a smaller version of the score --
    and AUC compares rows against each other.
    """
    return sum(w * d[c] for c, w in LATENT_W.items()).rename(f"latent{suffix}")


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """Adds the requested feature groups. Fit on the training fold only.

    Implements the fit_transform/transform contract that `run_cv(preprocessor=)`
    expects, so anything it learns from data is confined to the training fold.
    """

    def __init__(self, groups, seed: int = SEED, imputer_max_iter: int = 10):
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown group(s) {sorted(unknown)}; valid: {GROUPS}")
        self.groups = list(groups)
        self.seed = seed
        self.imputer_max_iter = imputer_max_iter

        # Anything below is LEARNED and must only see the training fold.
        self.imputer_ = None
        self.weekend_reg_ = None
        self.logit_ = None
        self.logit_scaler_ = None

    # -- internals ---------------------------------------------------------

    def _needs_imputer(self) -> bool:
        return bool(
            {"imputed", "latent", "weekend_resid", "imp_resid", "oof_logit"}
            & set(self.groups)
        )

    def _impute(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Return imputed numerics under their ORIGINAL names (for the formulas)."""
        if fit:
            self.imputer_ = IterativeImputer(
                max_iter=self.imputer_max_iter, random_state=self.seed
            )
            arr = self.imputer_.fit_transform(X[NUM_COLS])
        else:
            arr = self.imputer_.transform(X[NUM_COLS])
        return pd.DataFrame(arr, columns=NUM_COLS, index=X.index)

    def _build(self, X: pd.DataFrame, y=None, fit: bool = False) -> pd.DataFrame:
        Xi = self._impute(X, fit=fit) if self._needs_imputer() else None
        new = []

        if "imputed" in self.groups:
            new.append(Xi.add_suffix("_imp"))

        if "latent" in self.groups:
            new.append(_latent(Xi, "_imp").to_frame())
            new.append(_combos(Xi, "_imp"))
            new.append(_ratios(Xi, "_imp")[["social_ratio_imp", "leisure_ratio_imp"]])

        if "combos" in self.groups:
            new.append(_combos(X))

        if "ratios" in self.groups:
            new.append(_ratios(X))

        if "intensity" in self.groups:
            new.append(_intensity(X))

        if "weekend_resid" in self.groups:
            if fit:
                self.weekend_reg_ = LinearRegression().fit(Xi[[DAILY]], Xi[WEEKEND])
            resid = Xi[WEEKEND] - self.weekend_reg_.predict(Xi[[DAILY]])
            new.append(resid.rename("weekend_residual").to_frame())

        if "imp_resid" in self.groups:
            # |observed - what the other columns predicted|; NaN where unobserved.
            new.append(
                (X[NUM_COLS] - Xi[NUM_COLS]).abs().add_suffix("_resid")
            )

        if "oof_logit" in self.groups:
            new.append(self._logit_feature(Xi, y, fit).to_frame())

        if "n_missing" in self.groups:
            new.append(X[NUM_COLS].isna().sum(axis=1).rename("n_missing").to_frame())

        return pd.concat([X] + new, axis=1) if new else X.copy()

    def _logit_feature(self, Xi: pd.DataFrame, y, fit: bool) -> pd.Series:
        """Logistic-regression prediction as a feature ("stacking as a feature").

        On fit we use cross_val_predict so the training rows get OUT-OF-FOLD
        predictions -- otherwise the GBDT would see a logit that had already
        memorised that row's label and would over-trust it.
        """
        if fit:
            self.logit_scaler_ = StandardScaler().fit(Xi)
            Z = self.logit_scaler_.transform(Xi)
            self.logit_ = LogisticRegression(max_iter=3000)
            oof = cross_val_predict(self.logit_, Z, y, cv=5, method="predict_proba")[:, 1]
            self.logit_.fit(Z, y)
            return pd.Series(oof, index=Xi.index, name="logit_pred")
        Z = self.logit_scaler_.transform(Xi)
        return pd.Series(
            self.logit_.predict_proba(Z)[:, 1], index=Xi.index, name="logit_pred"
        )

    # -- sklearn-ish API ---------------------------------------------------

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self._build(X, y=y, fit=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build(X, y=None, fit=False)

    def fit(self, X: pd.DataFrame, y=None):
        self._build(X, y=y, fit=True)
        return self


def make_feature_pipeline(groups):
    """Factory for run_cv(preprocessor=...), which calls it once per fold.

        run_cv(mk_model, X, y, X_test,
               preprocessor=make_feature_pipeline(["imputed", "latent"]))
    """
    return lambda: FeaturePipeline(groups)


def add_group(X: pd.DataFrame, group: str, y=None) -> pd.DataFrame:
    """Apply one group outside the CV loop -- for inspection only.

    Do NOT use this to build a training set: groups that learn from data (the
    imputer, weekend_resid, oof_logit) would see every row and leak. Use
    `make_feature_pipeline` with run_cv/quick_cv for anything you will score.
    """
    return FeaturePipeline([group]).fit_transform(X, y)
