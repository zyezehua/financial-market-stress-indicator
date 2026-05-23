"""
Specialized binary classifier: Down vs Not-Down market direction.

Motivation: the 3-class direction model consistently under-detects Down moves
(OOS Recall ~0.11–0.34) because Down events are rare (~20% of samples) and
the model defaults to the majority class. A dedicated binary model with
balanced weights and a tunable probability threshold fixes this.

Architecture
────────────
Each model is a binary classifier trained on:
  y = 1  if market_dir = -1 (Down)
  y = 0  otherwise (Neutral or Up)

We expose a `predict_proba_down()` method returning P(Down). The caller
(DownRiskEnsemble) combines these probabilities and applies a threshold.
"""

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

LGBM_DOWN_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_DOWN_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


class LGBMDownRiskModel:
    def __init__(self, params: dict = None):
        self.params = params or LGBM_DOWN_PARAMS
        self.model = LGBMClassifier(**self.params, class_weight="balanced")
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return (y == -1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMDownRiskModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, self._encode(y))
        logger.info("LGBMDownRiskModel fitted on %d samples", len(X))
        return self

    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class XGBDownRiskModel:
    def __init__(self, params: dict = None):
        self.params = params or XGB_DOWN_PARAMS
        self.model = XGBClassifier(**self.params)
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return (y == -1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBDownRiskModel":
        self.feature_names = list(X.columns)
        y_enc = self._encode(y)
        sample_weight = compute_sample_weight("balanced", y_enc)
        self.model.fit(X, y_enc, sample_weight=sample_weight)
        logger.info("XGBDownRiskModel fitted on %d samples", len(X))
        return self

    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class RidgeDownRiskModel:
    def __init__(self, C: float = 0.5):
        self.C = C
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        self.feature_names: list = []

    def _encode(self, y: pd.Series) -> np.ndarray:
        return (y == -1).astype(int).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeDownRiskModel":
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, self._encode(y))
        logger.info("RidgeDownRiskModel fitted on %d samples", len(X))
        return self

    def predict_proba_down(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0))
        return self.model.predict_proba(X_scaled)[:, 1]
