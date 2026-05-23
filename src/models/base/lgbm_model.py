"""LightGBM wrapper for stress score, stress class, and direction prediction."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

logger = logging.getLogger(__name__)

LGBM_STRESS_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

LGBM_DIR_PARAMS = {
    **LGBM_STRESS_PARAMS,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
}


class LGBMStressModel:
    """Predicts continuous stress score (regression)."""

    def __init__(self, params: dict = None):
        self.params = params or LGBM_STRESS_PARAMS
        self.model = LGBMRegressor(**self.params)
        self.feature_names: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMStressModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        logger.info("LGBMStressModel fitted on %d samples, %d features", len(X), len(self.feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[self.feature_names]), 0, 100)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def booster_(self):
        return self.model.booster_


LGBM_STRESS_CLASS_PARAMS = {
    **LGBM_STRESS_PARAMS,
    "n_estimators": 600,
    "num_leaves": 31,
    "min_child_samples": 10,
}

STRESS_CLASS_LABELS = {"Low": 0, "Elevated": 1, "High": 2, "Extreme": 3}
STRESS_CLASS_REVERSE = {0: "Low", 1: "Elevated", 2: "High", 3: "Extreme"}


class LGBMStressClassModel:
    """Directly predicts stress class (Low/Elevated/High/Extreme) with balanced class weights."""

    def __init__(self, params: dict = None):
        self.params = params or LGBM_STRESS_CLASS_PARAMS
        filtered = {
            k: v for k, v in self.params.items()
            if k not in ("objective", "num_class", "metric", "n_jobs", "verbose")
        }
        self.model = LGBMClassifier(
            **filtered, objective="multiclass", num_class=4,
            class_weight="balanced", n_jobs=-1, verbose=-1,
        )
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return y.map(STRESS_CLASS_LABELS).fillna(1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMStressClassModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, self._encode(y))
        logger.info("LGBMStressClassModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.model.predict(X[self.feature_names])
        return np.array([STRESS_CLASS_REVERSE[e] for e in encoded])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class LGBMDirectionModel:
    """Predicts market direction: -1 (Down), 0 (Neutral), 1 (Up) as a 3-class classifier."""

    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    REVERSE_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, params: dict = None):
        self.params = params or LGBM_DIR_PARAMS
        # Exclude keys we set explicitly to avoid duplicate keyword argument errors
        filtered = {
            k: v for k, v in self.params.items()
            if k not in ("objective", "num_class", "metric", "n_jobs", "verbose")
        }
        self.model = LGBMClassifier(
            **filtered, objective="multiclass", num_class=3,
            class_weight="balanced", n_jobs=-1, verbose=-1,
        )
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return y.map(self.LABEL_MAP).fillna(1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMDirectionModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, self._encode(y))
        logger.info("LGBMDirectionModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.model.predict(X[self.feature_names])
        return np.array([self.REVERSE_MAP[e] for e in encoded])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
