"""LightGBM point forecast + quantile regression for prediction intervals."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import QUANTILES

log = logging.getLogger(__name__)

PARAMS = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
MAX_TREES = 2000
EARLY_STOP = 100

# name -> (objective, alpha)
SPECS = {
    "point": ("regression", None),
    "lo": ("quantile", QUANTILES[0]),
    "hi": ("quantile", QUANTILES[1]),
}


def fit_lgbm(X, y, X_val=None, y_val=None, objective="regression", alpha=None, n_estimators=None):
    """Fit one LightGBM model. With a validation set -> early stopping; otherwise fixed
    ``n_estimators`` (used for the final refit on all data)."""
    import lightgbm as lgb

    params = dict(PARAMS, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    if X_val is not None:
        model = lgb.LGBMRegressor(n_estimators=n_estimators or MAX_TREES, **params)
        model.fit(X, y, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    else:
        model = lgb.LGBMRegressor(n_estimators=n_estimators or 500, **params)
        model.fit(X, y)
    return model


def fit_model_set(train: pd.DataFrame, valid: pd.DataFrame | None, feats: list[str],
                  n_estimators: int | None = None, target: str = "y_res") -> dict:
    """Fit the point model and the two quantile models (5 % / 95 % -> 90 % interval).
    The target is the *residual* ``y - ref`` (ref = median of the same hour in the previous 4 weeks)."""
    models = {}
    for name, (obj, alpha) in SPECS.items():
        models[name] = fit_lgbm(
            train[feats], train[target],
            valid[feats] if valid is not None else None,
            valid[target] if valid is not None else None,
            objective=obj, alpha=alpha, n_estimators=n_estimators,
        )
        log.info("  fitted %-5s (best_iter=%s)", name, getattr(models[name], "best_iteration_", None))
    return models


def predict_set(models: dict, X: pd.DataFrame, ref) -> dict[str, np.ndarray]:
    """Predict point + raw quantile interval in MW (residual + ref) and repair quantile crossing."""
    ref = np.asarray(ref, dtype=float)
    point = ref + models["point"].predict(X)
    lo, hi = ref + models["lo"].predict(X), ref + models["hi"].predict(X)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return {"point": point, "lo": np.minimum(lo, point), "hi": np.maximum(hi, point)}


# ---------------------------------------------------------------- conformal calibration (CQR)
def _bin_key(a, b): return f"{a}-{b}"


def conformal_offsets(y, lo, hi, horizon, alpha: float, bins) -> dict[str, float]:
    """Conformalised quantile regression (Romano et al. 2019): on a held-out calibration set,
    compute how far the truth falls outside [lo, hi] and take the (1-alpha) quantile of those
    scores. Adding it to both sides gives finite-sample ~(1-alpha) coverage (if calibration and test
    data are exchangeable - roughly true here, not exactly). Estimated per horizon group."""
    y, lo, hi, horizon = map(np.asarray, (y, lo, hi, horizon))
    scores = np.maximum(lo - y, y - hi)

    def q(s):
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(s, level, method="higher"))

    overall = q(scores)
    out = {}
    for a, b in bins:
        m = (horizon >= a) & (horizon <= b)
        out[_bin_key(a, b)] = q(scores[m]) if m.sum() >= 50 else overall
    return out


def apply_conformal(lo, hi, horizon, offsets: dict[str, float]):
    lo, hi, horizon = np.array(lo, float), np.array(hi, float), np.asarray(horizon)
    for key, off in offsets.items():
        a, b = map(int, key.split("-"))
        m = (horizon >= a) & (horizon <= b)
        lo[m] -= off
        hi[m] += off
    return lo, hi
