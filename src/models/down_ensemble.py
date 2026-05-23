"""
DownRiskEnsemble: hierarchical direction predictor.

Logic
─────
1. DownRiskEnsemble blends P(Down) from three binary Down specialists.
2. When P(Down) >= down_threshold, the final prediction is "Down" regardless
   of what the main 3-class DirectionEnsemble says.
3. Otherwise the main DirectionEnsemble result is used.

This two-stage approach preserves Neutral/Up accuracy while significantly
improving Down recall — the main gap identified in Phase 1 OOS results.
"""

import logging

import numpy as np
import pandas as pd

from .base.down_risk_model import LGBMDownRiskModel, XGBDownRiskModel, RidgeDownRiskModel
from .ensemble import DirectionEnsemble

logger = logging.getLogger(__name__)

DEFAULT_DOWN_WEIGHTS = {"lgbm": 0.5, "xgb": 0.3, "ridge": 0.2}
DEFAULT_DOWN_THRESHOLD = 0.40   # P(Down) >= 40% triggers Down override


class DownRiskEnsemble:
    """
    Wraps a DirectionEnsemble and overrides its output when the Down-specialist
    ensemble signals elevated downside probability.

    Parameters
    ----------
    lgbm_down, xgb_down, ridge_down : fitted binary Down-risk models
    direction_ens                    : fitted DirectionEnsemble (3-class)
    weights                          : blending weights for binary models
    down_threshold                   : P(Down) cutoff for override (default 0.40)
    """

    DIRECTION_LABELS = {-1: "Down", 0: "Neutral", 1: "Up"}

    def __init__(
        self,
        lgbm_down: LGBMDownRiskModel,
        xgb_down: XGBDownRiskModel,
        ridge_down: RidgeDownRiskModel,
        direction_ens: DirectionEnsemble,
        weights: dict = None,
        down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    ):
        self.binary_models = {
            "lgbm": lgbm_down,
            "xgb": xgb_down,
            "ridge": ridge_down,
        }
        self.direction_ens = direction_ens
        self.weights = weights or DEFAULT_DOWN_WEIGHTS
        self.down_threshold = down_threshold

    def predict_down_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Blended P(Down) from all binary specialists."""
        proba = np.zeros(len(X))
        total_w = 0.0
        for name, model in self.binary_models.items():
            w = self.weights.get(name, 0.0)
            if w == 0 or model is None:
                continue
            proba += w * model.predict_proba_down(X)
            total_w += w
        if total_w > 0:
            proba /= total_w
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns integer direction array (-1 / 0 / 1).
        Down override applied where P(Down) >= down_threshold.
        """
        main_pred   = self.direction_ens.predict(X)       # (-1, 0, 1) array
        down_proba  = self.predict_down_proba(X)
        override    = down_proba >= self.down_threshold
        final       = np.where(override, -1, main_pred)
        n_overrides = int(override.sum())
        if n_overrides:
            logger.debug("Down override applied to %d / %d rows", n_overrides, len(X))
        return final

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (N, 3) probability matrix [P(Down), P(Neutral), P(Up)].
        When Down override fires, we redistribute probability mass toward Down.
        """
        main_proba  = self.direction_ens.predict_proba(X)   # (N, 3): Down, Neutral, Up
        down_proba  = self.predict_down_proba(X)             # (N,)
        override    = down_proba >= self.down_threshold      # (N,)

        blended = main_proba.copy()
        if override.any():
            # For overridden rows: set P(Down) = max(main, specialist), renormalize
            blended[override, 0] = np.maximum(
                main_proba[override, 0], down_proba[override]
            )
            row_sums = blended[override].sum(axis=1, keepdims=True)
            blended[override] /= np.where(row_sums > 0, row_sums, 1)

        return blended
