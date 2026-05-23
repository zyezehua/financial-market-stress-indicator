"""
Multi-horizon inference: runs all trained ensembles on the latest feature row.

Output for each horizon:
    stress_score, stress_class, stress_delta, market_direction, market_dir_proba
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .trainer import load_artifact

logger = logging.getLogger(__name__)

DIRECTION_LABELS = {-1: "Down", 0: "Neutral", 1: "Up"}

STRESS_CLASSES = {
    (0,  25): "Low",
    (26, 50): "Elevated",
    (51, 75): "High",
    (76, 100): "Extreme",
}


def _classify(score: float) -> str:
    for (lo, hi), label in STRESS_CLASSES.items():
        if lo <= score <= hi:
            return label
    return "Extreme"


def predict_latest(
    features: pd.DataFrame,
    csi: pd.DataFrame,
    horizons: List[int] = None,
    artifacts_dir: str = "models",
    as_of: Optional[str] = None,
) -> Dict[int, dict]:
    """
    Run inference on the most recent available feature row.

    Parameters
    ----------
    features     : full feature matrix (from builder.py)
    csi          : CSI DataFrame (from composite_index.py)
    horizons     : list of prediction horizons in days
    artifacts_dir: where model artifacts are stored
    as_of        : date string (YYYY-MM-DD) to simulate a past prediction date

    Returns
    -------
    Dict mapping horizon → prediction dict
    """
    horizons = horizons or [5, 21, 63]

    if as_of:
        row_date = pd.Timestamp(as_of)
        if row_date not in features.index:
            row_date = features.index[features.index <= row_date][-1]
    else:
        row_date = features.index[-1]

    X_row = features.loc[[row_date]]
    current_csi = float(csi.loc[row_date, "csi_composite"]) if row_date in csi.index else np.nan

    results = {}
    for h in horizons:
        try:
            artifact = load_artifact(h, artifacts_dir)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue

        feat_names = artifact["feature_names"]
        # Align feature columns
        X_aligned = X_row.reindex(columns=feat_names, fill_value=0)

        stress_ens = artifact["stress_ensemble"]
        stress_pred = float(stress_ens.predict(X_aligned)[0])
        stress_class = _classify(stress_pred)
        stress_delta = stress_pred - current_csi if not np.isnan(current_csi) else np.nan

        dir_result = {}
        dir_ens = artifact.get("direction_ensemble")
        if dir_ens is not None:
            dir_pred = int(dir_ens.predict(X_aligned)[0])
            dir_proba = dir_ens.predict_proba(X_aligned)[0]
            dir_result = {
                "market_direction":     DIRECTION_LABELS[dir_pred],
                "market_dir_raw":       dir_pred,
                "market_dir_proba":     {
                    "Down":    round(float(dir_proba[0]), 3),
                    "Neutral": round(float(dir_proba[1]), 3),
                    "Up":      round(float(dir_proba[2]), 3),
                },
            }

        results[h] = {
            "as_of":          row_date.strftime("%Y-%m-%d"),
            "horizon_days":   h,
            "current_csi":    round(current_csi, 1) if not np.isnan(current_csi) else None,
            "stress_score":   round(stress_pred, 1),
            "stress_class":   stress_class,
            "stress_delta":   round(stress_delta, 1) if not np.isnan(stress_delta) else None,
            **dir_result,
        }
        logger.info(
            "Horizon %dd | Stress: %.1f (%s) | ΔStress: %+.1f | Direction: %s",
            h,
            stress_pred,
            stress_class,
            stress_delta if not np.isnan(stress_delta) else 0,
            dir_result.get("market_direction", "N/A"),
        )

    return results


def predict_historical(
    features: pd.DataFrame,
    csi: pd.DataFrame,
    horizon: int,
    start: Optional[str] = None,
    end: Optional[str] = None,
    artifacts_dir: str = "models",
) -> pd.DataFrame:
    """
    Generate historical predictions for backtesting or report charts.

    Returns DataFrame with stress_score_pred, stress_class_pred, stress_actual columns.
    """
    artifact = load_artifact(horizon, artifacts_dir)
    feat_names = artifact["feature_names"]
    stress_ens = artifact["stress_ensemble"]

    feat_slice = features.copy()
    if start:
        feat_slice = feat_slice.loc[start:]
    if end:
        feat_slice = feat_slice.loc[:end]

    X = feat_slice.reindex(columns=feat_names, fill_value=0)
    preds = stress_ens.predict(X)

    result = pd.DataFrame({
        "stress_score_pred": preds,
        "stress_class_pred": [_classify(p) for p in preds],
    }, index=feat_slice.index)

    if "csi_composite" in csi.columns:
        result["stress_actual"] = csi["csi_composite"].reindex(feat_slice.index)

    return result
