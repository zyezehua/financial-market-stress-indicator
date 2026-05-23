"""Weighted ensemble of LightGBM, XGBoost, and Ridge models."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"lgbm": 0.5, "xgb": 0.3, "ridge": 0.2}


class StressEnsemble:
    """Weighted average ensemble for stress score regression."""

    def __init__(self, lgbm, xgb, ridge, weights: dict = None):
        self.models = {"lgbm": lgbm, "xgb": xgb, "ridge": ridge}
        self.weights = weights or DEFAULT_WEIGHTS

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        total_w = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w == 0 or model is None:
                continue
            preds += w * model.predict(X)
            total_w += w
        if total_w > 0:
            preds /= total_w
        return np.clip(preds, 0, 100)

    def feature_importances(self, feature_names: list) -> pd.Series:
        """Average importance across tree-based models (Ridge excluded)."""
        imps = {}
        for name in ["lgbm", "xgb"]:
            m = self.models.get(name)
            if m is None:
                continue
            fi = pd.Series(m.feature_importances_, index=feature_names)
            imps[name] = fi / fi.sum()
        if not imps:
            return pd.Series(dtype=float)
        avg = pd.concat(imps.values(), axis=1).mean(axis=1)
        return avg.sort_values(ascending=False)


class DirectionEnsemble:
    """
    Soft-voting ensemble for 3-class direction prediction.
    Probabilities are averaged across models before argmax.
    """

    REVERSE_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, lgbm, xgb, ridge, weights: dict = None):
        self.models = {"lgbm": lgbm, "xgb": xgb, "ridge": ridge}
        self.weights = weights or DEFAULT_WEIGHTS

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (N, 3) probability matrix."""
        proba = np.zeros((len(X), 3))
        total_w = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w == 0 or model is None:
                continue
            proba += w * model.predict_proba(X)
            total_w += w
        if total_w > 0:
            proba /= total_w
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        encoded = np.argmax(proba, axis=1)
        return np.array([self.REVERSE_MAP[e] for e in encoded])
