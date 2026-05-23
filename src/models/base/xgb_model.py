"""XGBoost wrapper for stress and direction prediction."""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)

XGB_STRESS_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}

XGB_DIR_PARAMS = {
    **XGB_STRESS_PARAMS,
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
}


class XGBStressModel:
    def __init__(self, params: dict = None):
        self.params = params or XGB_STRESS_PARAMS
        self.model = XGBRegressor(**self.params)
        self.feature_names: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBStressModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        logger.info("XGBStressModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X[self.feature_names]), 0, 100)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class XGBDirectionModel:
    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    REVERSE_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, params: dict = None):
        self.params = params or XGB_DIR_PARAMS
        self.model = XGBClassifier(**self.params)
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return y.map(self.LABEL_MAP).fillna(1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBDirectionModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, self._encode(y))
        logger.info("XGBDirectionModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.model.predict(X[self.feature_names])
        return np.array([self.REVERSE_MAP[e] for e in encoded])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
