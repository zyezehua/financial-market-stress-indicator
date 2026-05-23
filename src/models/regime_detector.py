"""
HMM-based market regime detector.

Identifies 3 hidden regimes from the CSI composite history:
    Regime 0 — Low Stress     (CSI typically 0-30)
    Regime 1 — Moderate Stress (CSI typically 25-60)
    Regime 2 — High/Crisis    (CSI typically 55-100)

The regimes are used to apply regime-conditioned CSI dimension weights,
so the index is more sensitive to the right signals in each environment.

Why HMM?
────────
A simple threshold classifier would misfire at regime boundaries.
HMM adds temporal persistence (stress regimes tend to last weeks/months),
so the model only switches regime when the evidence is sustained.
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn not installed. RegimeDetector will fall back to threshold-based "
        "regime classification. Install with: pip install hmmlearn"
    )

N_REGIMES = 3
MIN_HISTORY = 252   # need at least 1 year to fit HMM


def _threshold_regimes(csi: pd.Series) -> np.ndarray:
    """Simple fallback: assign regime by fixed CSI thresholds."""
    regimes = np.ones(len(csi), dtype=int)   # default: Moderate
    regimes[csi.values < 30] = 0             # Low Stress
    regimes[csi.values > 55] = 2             # High/Crisis
    return regimes


class RegimeDetector:
    """
    Fits a Gaussian HMM on CSI composite returns and assigns each day
    to one of N_REGIMES regimes, ordered by mean CSI level.

    Parameters
    ----------
    n_regimes    : number of hidden states (default 3)
    n_iter       : HMM EM iterations
    artifacts_dir: where to save/load the fitted HMM
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        n_iter: int = 200,
        artifacts_dir: str = "models",
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.artifacts_dir = artifacts_dir
        self.model: Optional[object] = None
        self._regime_order: Optional[np.ndarray] = None   # maps raw HMM state → ordered index

    def fit(self, csi_composite: pd.Series) -> "RegimeDetector":
        """
        Fit HMM on CSI composite levels + 21-day change as observation features.

        The two-feature observation makes the HMM sensitive to both the level
        (are we stressed?) and the direction (are we entering or leaving stress?).
        """
        series = csi_composite.dropna()
        if len(series) < MIN_HISTORY:
            raise ValueError(f"Need >= {MIN_HISTORY} observations to fit HMM.")

        level = series.values.reshape(-1, 1)
        change_21d = pd.Series(series.values).diff(21).fillna(0).values.reshape(-1, 1)
        X = np.hstack([level, change_21d])

        if not HMM_AVAILABLE:
            logger.warning("Using threshold fallback — hmmlearn not available.")
            self.model = None
            return self

        hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
        )
        hmm.fit(X)
        self.model = hmm

        # Order states by mean CSI level (state 0 = lowest stress)
        raw_states = hmm.predict(X)
        means = np.array([level[raw_states == s].mean() for s in range(self.n_regimes)])
        self._regime_order = np.argsort(means)   # index i → ordered rank

        logger.info(
            "HMM fitted: %d regimes, mean CSI by regime: %s",
            self.n_regimes,
            {i: round(float(means[self._regime_order[i]]), 1) for i in range(self.n_regimes)},
        )
        return self

    def predict(self, csi_composite: pd.Series) -> pd.Series:
        """
        Assign a regime label (0=Low, 1=Moderate, 2=High) to each date.

        Returns a pd.Series aligned to csi_composite.index.
        """
        series = csi_composite.dropna()

        if not HMM_AVAILABLE or self.model is None:
            regimes = _threshold_regimes(series)
            return pd.Series(regimes, index=series.index, name="regime").reindex(csi_composite.index)

        level = series.values.reshape(-1, 1)
        change_21d = pd.Series(series.values).diff(21).fillna(0).values.reshape(-1, 1)
        X = np.hstack([level, change_21d])

        raw_states  = self.model.predict(X)
        # Map raw HMM state → ordered regime (0=Low, 1=Moderate, 2=High)
        state_to_ordered = {
            int(self._regime_order[i]): i for i in range(self.n_regimes)
        }
        ordered = np.array([state_to_ordered[s] for s in raw_states])
        return pd.Series(ordered, index=series.index, name="regime").reindex(csi_composite.index)

    def current_regime(self, csi_composite: pd.Series) -> Tuple[int, str]:
        """Return the most recent regime index and name."""
        regimes = self.predict(csi_composite)
        latest = int(regimes.dropna().iloc[-1])
        names = {0: "Low Stress", 1: "Moderate Stress", 2: "High/Crisis"}
        return latest, names.get(latest, "Unknown")

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.artifacts_dir, "regime_detector.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "regime_order": self._regime_order}, f)
        logger.info("RegimeDetector saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str, **kwargs) -> "RegimeDetector":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(**kwargs)
        obj.model = state["model"]
        obj._regime_order = state["regime_order"]
        return obj


# ── Regime-conditioned CSI dimension weights ──────────────────────────────────

REGIME_WEIGHTS = {
    # Low Stress: all dimensions contribute roughly equally
    0: {
        "equity":        0.18,
        "credit":        0.18,
        "rates":         0.18,
        "liquidity":     0.18,
        "fx":            0.14,
        "international": 0.14,
    },
    # Moderate Stress: equity and credit more important
    1: {
        "equity":        0.22,
        "credit":        0.22,
        "rates":         0.17,
        "liquidity":     0.17,
        "fx":            0.12,
        "international": 0.10,
    },
    # High/Crisis: liquidity and credit dominate (systemic risk channels)
    2: {
        "equity":        0.20,
        "credit":        0.25,
        "rates":         0.15,
        "liquidity":     0.22,
        "fx":            0.10,
        "international": 0.08,
    },
}


def get_regime_weights(regime: int) -> dict:
    """Return the CSI dimension weights for a given regime index."""
    return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[1])
