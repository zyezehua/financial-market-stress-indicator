"""
Model trainer: expanding-window cross-validation and monthly retraining.

For each horizon h, trains:
    StressEnsemble      → continuous stress score (regression)
    StressClassEnsemble → direct stress class with balanced weights (4-class)
    DirectionEnsemble   → 3-class market direction with balanced weights
    DownRiskEnsemble    → hierarchical direction with binary Down specialist
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .base.lgbm_model import LGBMStressModel, LGBMDirectionModel, LGBMStressClassModel
from .base.ridge_model import RidgeStressModel, RidgeDirectionModel, RidgeStressClassModel
from .base.xgb_model import XGBStressModel, XGBDirectionModel, XGBStressClassModel
from .base.down_risk_model import LGBMDownRiskModel, XGBDownRiskModel, RidgeDownRiskModel
from .ensemble import StressEnsemble, DirectionEnsemble, StressClassEnsemble
from .down_ensemble import DownRiskEnsemble

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
    csi_composite: pd.Series = None,
) -> Dict[str, object]:
    """
    Train all ensembles for a single prediction horizon.

    Returns a dict with:
        stress_ensemble, stress_class_ensemble, direction_ensemble,
        down_risk_ensemble, feature_names, stress_metrics, direction_metrics
    """
    logger.info("Training models for horizon %dd ...", horizon)

    stress_col = f"stress_fwd_{horizon}d"
    market_col = f"market_dir_{horizon}d"
    class_col  = f"stress_class_fwd_{horizon}d"   # derived below from stress_fwd

    if stress_col not in targets.columns:
        raise ValueError(f"Target column '{stress_col}' not found.")

    X = features.copy()
    y_stress = targets[stress_col]
    y_dir    = targets.get(market_col)

    X_s, y_s = _drop_missing(X, y_stress)
    if len(X_s) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Only {len(X_s)} clean samples for horizon {horizon}d "
            f"(minimum {MIN_TRAIN_SAMPLES} required)."
        )

    # Derive stress class labels using training-data quantiles (regime-adaptive)
    cls_q30, cls_q50, cls_q70 = [
        float(np.nanpercentile(y_s.values, p)) for p in [30, 50, 70]
    ]
    logger.info(
        "Class thresholds (h=%dd): Low≤%.1f  Elevated≤%.1f  High≤%.1f  Extreme>%.1f",
        horizon, cls_q30, cls_q50, cls_q70, cls_q70,
    )

    def _to_class(v):
        if pd.isna(v): return np.nan
        if v <= cls_q30: return "Low"
        if v <= cls_q50: return "Elevated"
        if v <= cls_q70: return "High"
        return "Extreme"

    y_stress_class = y_stress.apply(_to_class).dropna()
    X_sc = X.loc[y_stress_class.index].dropna()
    y_sc = y_stress_class.loc[X_sc.index]

    # ── Expanding window CV for stress regression diagnostics ─────────────
    tscv = TimeSeriesSplit(n_splits=5)
    stress_val_errors = []
    for train_idx, val_idx in tscv.split(X_s):
        if len(train_idx) < MIN_TRAIN_SAMPLES // 2:
            continue
        X_tr, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
        y_tr, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]
        lgbm_s = LGBMStressModel().fit(X_tr, y_tr)
        preds  = lgbm_s.predict(X_val)
        stress_val_errors.append(float(np.mean(np.abs(preds - y_val.values))))

    avg_mae = float(np.mean(stress_val_errors)) if stress_val_errors else None
    logger.info("Stress CV MAE (horizon %dd): %.2f", horizon, avg_mae or -1)

    # ── Train stress regression ensemble ──────────────────────────────────
    lgbm_stress  = LGBMStressModel().fit(X_s, y_s)
    xgb_stress   = XGBStressModel().fit(X_s, y_s)
    ridge_stress  = RidgeStressModel().fit(X_s, y_s)
    stress_ens   = StressEnsemble(lgbm_stress, xgb_stress, ridge_stress, weights)

    # ── Train stress class ensemble (balanced weights) ─────────────────────
    stress_class_ens = None
    if len(X_sc) >= MIN_TRAIN_SAMPLES:
        lgbm_sc  = LGBMStressClassModel().fit(X_sc, y_sc)
        xgb_sc   = XGBStressClassModel().fit(X_sc, y_sc)
        ridge_sc  = RidgeStressClassModel().fit(X_sc, y_sc)
        stress_class_ens = StressClassEnsemble(lgbm_sc, xgb_sc, ridge_sc, weights)
        logger.info("StressClassEnsemble trained for horizon %dd.", horizon)

    # ── Train direction ensembles ─────────────────────────────────────────
    direction_ens  = None
    down_risk_ens  = None
    dir_metrics    = {}

    if y_dir is not None:
        X_d, y_d = _drop_missing(X, y_dir)
        if len(X_d) >= MIN_TRAIN_SAMPLES:
            # 3-class direction (balanced)
            lgbm_dir  = LGBMDirectionModel().fit(X_d, y_d)
            xgb_dir   = XGBDirectionModel().fit(X_d, y_d)
            ridge_dir = RidgeDirectionModel().fit(X_d, y_d)
            direction_ens = DirectionEnsemble(lgbm_dir, xgb_dir, ridge_dir, weights)

            # Binary Down-risk specialists
            lgbm_dn  = LGBMDownRiskModel().fit(X_d, y_d)
            xgb_dn   = XGBDownRiskModel().fit(X_d, y_d)
            ridge_dn = RidgeDownRiskModel().fit(X_d, y_d)
            down_risk_ens = DownRiskEnsemble(
                lgbm_dn, xgb_dn, ridge_dn,
                direction_ens=direction_ens,
                weights=weights,
            )

            # Direction accuracy on last 500 OOS-ish samples
            preds_dir = down_risk_ens.predict(X_d.tail(500))
            actual    = y_d.tail(500).values
            acc = float(np.mean(preds_dir == actual))
            dir_metrics = {"direction_accuracy_last_500": round(acc, 3)}
            logger.info("Direction accuracy (horizon %dd, last 500d): %.1f%%", horizon, acc * 100)

    # ── Fit regime detector (if CSI composite provided) ───────────────────
    from .regime_detector import RegimeDetector
    regime_detector = None
    if csi_composite is not None:
        try:
            rd = RegimeDetector()
            rd.fit(csi_composite)
            regime_detector = rd
            logger.info("RegimeDetector fitted for horizon %dd.", horizon)
        except Exception as exc:
            logger.warning("RegimeDetector fit failed: %s", exc)

    # ── Persist artifact ──────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    artifact = {
        "horizon":              horizon,
        "trained_at":           datetime.now().isoformat(),
        "train_data_end":       X_s.index.max().isoformat(),
        "stress_ensemble":      stress_ens,
        "stress_class_ensemble": stress_class_ens,
        "direction_ensemble":   direction_ens,
        "down_risk_ensemble":   down_risk_ens,
        "feature_names":        list(X_s.columns),
        "stress_metrics":       {"cv_mae": avg_mae},
        "direction_metrics":    dir_metrics,
        "class_thresholds":     {"q30": cls_q30, "q50": cls_q50, "q70": cls_q70},
        "regime_detector":      regime_detector,
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
    csi_composite: pd.Series = None,
) -> Dict[int, dict]:
    horizons = horizons or [5, 21, 63]
    results = {}
    for h in horizons:
        try:
            results[h] = train_horizon(
                features, targets, h, weights, artifacts_dir,
                csi_composite=csi_composite,
            )
        except Exception as exc:
            logger.error("Failed to train horizon %dd: %s", h, exc)
    return results


def load_artifact(horizon: int, artifacts_dir: str = "models") -> dict:
    path = os.path.join(artifacts_dir, f"model_h{horizon}d.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}. Run training first.")
    with open(path, "rb") as f:
        return pickle.load(f)
