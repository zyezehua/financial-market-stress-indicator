"""
Model trainer: expanding-window cross-validation and monthly retraining.

For each horizon h, trains:
    - StressEnsemble   → predicts CSI composite h days ahead
    - DirectionEnsemble → predicts SPY market direction h days ahead
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .base.lgbm_model import LGBMStressModel, LGBMDirectionModel
from .base.ridge_model import RidgeStressModel, RidgeDirectionModel
from .base.xgb_model import XGBStressModel, XGBDirectionModel
from .ensemble import StressEnsemble, DirectionEnsemble

logger = logging.getLogger(__name__)

MIN_TRAIN_SAMPLES = 756   # 3 years


def _drop_missing(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]


def train_horizon(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
    weights: dict = None,
    artifacts_dir: str = "models",
) -> Dict[str, object]:
    """
    Train a StressEnsemble and DirectionEnsemble for a single prediction horizon.

    Uses an expanding window: trains on all data up to the cutoff,
    consistent with how we'll retrain monthly in production.

    Returns a dict with:
        stress_ensemble, direction_ensemble, stress_metrics, direction_metrics
    """
    logger.info("Training models for horizon %dd ...", horizon)

    stress_col  = f"stress_fwd_{horizon}d"
    market_col  = f"market_dir_{horizon}d"

    if stress_col not in targets.columns:
        raise ValueError(f"Target column '{stress_col}' not found in targets DataFrame.")

    X = features.copy()
    y_stress = targets[stress_col]
    y_dir    = targets.get(market_col)

    X_s, y_s = _drop_missing(X, y_stress)

    if len(X_s) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Only {len(X_s)} clean samples for horizon {horizon}d "
            f"(minimum {MIN_TRAIN_SAMPLES} required)."
        )

    # ── Expanding window CV for diagnostics ──────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    stress_val_errors = []
    for train_idx, val_idx in tscv.split(X_s):
        if len(train_idx) < MIN_TRAIN_SAMPLES // 2:
            continue
        X_tr, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
        y_tr, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]

        lgbm_s = LGBMStressModel().fit(X_tr, y_tr)
        preds  = lgbm_s.predict(X_val)
        mae    = float(np.mean(np.abs(preds - y_val.values)))
        stress_val_errors.append(mae)

    avg_mae = float(np.mean(stress_val_errors)) if stress_val_errors else None
    logger.info("Stress CV MAE (horizon %dd): %.2f", horizon, avg_mae or -1)

    # ── Final training on all available data ──────────────────────────────
    lgbm_stress  = LGBMStressModel().fit(X_s, y_s)
    xgb_stress   = XGBStressModel().fit(X_s, y_s)
    ridge_stress  = RidgeStressModel().fit(X_s, y_s)
    stress_ens   = StressEnsemble(lgbm_stress, xgb_stress, ridge_stress, weights)

    direction_ens = None
    dir_metrics   = {}
    if y_dir is not None:
        X_d, y_d = _drop_missing(X, y_dir)
        if len(X_d) >= MIN_TRAIN_SAMPLES:
            lgbm_dir  = LGBMDirectionModel().fit(X_d, y_d)
            xgb_dir   = XGBDirectionModel().fit(X_d, y_d)
            ridge_dir = RidgeDirectionModel().fit(X_d, y_d)
            direction_ens = DirectionEnsemble(lgbm_dir, xgb_dir, ridge_dir, weights)

            preds_dir = direction_ens.predict(X_d.tail(500))
            actual    = y_d.tail(500).values
            acc = float(np.mean(preds_dir == actual))
            dir_metrics = {"direction_accuracy_last_500": round(acc, 3)}
            logger.info("Direction accuracy (horizon %dd, last 500d): %.1f%%", horizon, acc * 100)

    # ── Persist artifacts ─────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    artifact = {
        "horizon":          horizon,
        "trained_at":       datetime.now().isoformat(),
        "stress_ensemble":  stress_ens,
        "direction_ensemble": direction_ens,
        "feature_names":    list(X_s.columns),
        "stress_metrics":   {"cv_mae": avg_mae},
        "direction_metrics": dir_metrics,
    }
    path = os.path.join(artifacts_dir, f"model_h{horizon}d.pkl")
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved model artifact: %s", path)

    return artifact


def train_all_horizons(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizons: List[int] = None,
    weights: dict = None,
    artifacts_dir: str = "models",
) -> Dict[int, dict]:
    horizons = horizons or [5, 21, 63]
    results = {}
    for h in horizons:
        try:
            results[h] = train_horizon(features, targets, h, weights, artifacts_dir)
        except Exception as exc:
            logger.error("Failed to train horizon %dd: %s", h, exc)
    return results


def load_artifact(horizon: int, artifacts_dir: str = "models") -> dict:
    path = os.path.join(artifacts_dir, f"model_h{horizon}d.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}. Run training first.")
    with open(path, "rb") as f:
        return pickle.load(f)
