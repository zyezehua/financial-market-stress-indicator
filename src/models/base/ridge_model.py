"""Ridge regression baseline for stress prediction."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RidgeStressModel:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_names: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeStressModel":
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        logger.info("RidgeStressModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0))
        return np.clip(self.model.predict(X_scaled), 0, 100)

    @property
    def feature_importances_(self):
        return np.abs(self.model.coef_)


class RidgeDirectionModel:
    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    REVERSE_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, C: float = 0.1):
        self.C = C
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=1000, n_jobs=-1)
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return y.map(self.LABEL_MAP).fillna(1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeDirectionModel":
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, self._encode(y))
        logger.info("RidgeDirectionModel fitted on %d samples", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0))
        encoded = self.model.predict(X_scaled)
        return np.array([self.REVERSE_MAP[e] for e in encoded])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0))
        return self.model.predict_proba(X_scaled)
