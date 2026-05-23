"""LightGBM wrapper for stress and direction prediction."""

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


class LGBMDirectionModel:
    """Predicts market direction: -1 (Down), 0 (Neutral), 1 (Up) as a 3-class classifier."""

    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    REVERSE_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, params: dict = None):
        self.params = params or LGBM_DIR_PARAMS
        self.model = LGBMClassifier(**{
            k: v for k, v in self.params.items()
            if k not in ("objective", "num_class", "metric")
        }, objective="multiclass", num_class=3, n_jobs=-1, verbose=-1)
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
