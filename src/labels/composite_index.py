"""
Composite Stress Index (CSI) — self-built, 6-dimension weighted stress score.

Construction
------------
For each dimension d:
    1. Compute a raw dimension score (weighted avg of signals within dimension)
    2. Normalize to 0-100 via rolling percentile rank (3-year window)

Final CSI = weighted average of 6 normalized dimension scores.
"""

import logging
import os

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "equity":        0.20,
    "credit":        0.20,
    "rates":         0.175,
    "liquidity":     0.175,
    "fx":            0.125,
    "international": 0.125,
}

# Maps each dimension to its most representative "high stress = high value" features
# Each entry: (column_name, invert) — invert=True means raw signal is low-stress when high
DIMENSION_SIGNALS = {
    "equity": [
        ("eq_vix_pct_rank_252d",            False),
        ("eq_vix_term_ratio_pct_rank_252d",  False),
        ("eq_spy_rvol_21d_pct_rank_252d",    False),
        ("eq_spy_drawdown_252d",             False),  # already 0-positive, higher=worse
    ],
    "credit": [
        ("cr_hy_oas_pct_rank_252d",          False),
        ("cr_hy_ig_spread_pct_rank_252d",    False),
        ("cr_hyg_rvol_21d_pct_rank_252d",    False),
        ("cr_hyg_drawdown_63d",              False),
    ],
    "rates": [
        ("rt_tlt_rvol_21d_pct_rank_252d",    False),
        ("rt_t10y_abs_change_5d_pct_rank_252d", False),
        ("rt_inversion_depth_pct_rank_252d", False),
    ],
    "liquidity": [
        ("lq_avg_cross_corr_21d_pct_rank_252d", False),
        ("lq_funding_spread_pct_rank_252d",  False),
        ("lq_spy_tlt_corr_21d_pct_rank_252d", False),
    ],
    "fx": [
        ("fx_vol_composite_pct_rank_252d",   False),
        ("fx_jpy_strength_pct_rank_252d",    False),
        ("fx_usd_pct_rank_252d",             False),
    ],
    "international": [
        ("intl_eem_rvol_pct_rank_252d",      False),
        ("intl_vol_composite_pct_rank_252d", False),
        ("intl_gld_pct_rank_252d",           False),
    ],
}

CLASSIFICATION = {
    "Low":      (0,   25),
    "Elevated": (26,  50),
    "High":     (51,  75),
    "Extreme":  (76, 100),
}


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _dimension_score(features: pd.DataFrame, dim: str) -> pd.Series:
    """Average available signals for one dimension into a 0-100 score."""
    signals = DIMENSION_SIGNALS.get(dim, [])
    cols = []
    for col, invert in signals:
        if col in features.columns:
            s = features[col].copy()
            # Signals already in percentile rank (0-100); drawdown signals need ranking
            if not col.endswith("_pct_rank_252d"):
                s = _percentile_rank(s, 252)
            if invert:
                s = 100 - s
            cols.append(s)
    if not cols:
        logger.warning("No signals found for dimension: %s", dim)
        return pd.Series(np.nan, index=features.index)
    return pd.concat(cols, axis=1).mean(axis=1)


def build_csi(
    features: pd.DataFrame,
    weights: dict = None,
    percentile_window: int = 756,
    labels_dir: str = "data/labels",
    save: bool = True,
    regime_detector=None,
) -> pd.DataFrame:
    """
    Build the Composite Stress Index from the feature matrix.

    Parameters
    ----------
    regime_detector : optional fitted RegimeDetector; when provided, dimension
                      weights are adjusted per-day based on the detected regime.

    Returns
    -------
    DataFrame with columns:
        csi_equity, csi_credit, csi_rates, csi_liquidity, csi_fx, csi_international
        csi_composite  : weighted average of dimension scores
        csi_class      : Low / Elevated / High / Extreme
        csi_regime     : 0/1/2 regime label (only present when regime_detector provided)
    """
    weights = weights or DEFAULT_WEIGHTS
    dim_scores = {}

    for dim in DEFAULT_WEIGHTS:
        score = _dimension_score(features, dim)
        dim_scores[f"csi_{dim}"] = score
        logger.info("Built CSI dimension '%s': %.1f%% non-null", dim, score.notna().mean() * 100)

    result = pd.DataFrame(dim_scores, index=features.index)

    # Weighted composite (regime-conditioned if detector provided)
    if regime_detector is not None:
        try:
            from src.models.regime_detector import get_regime_weights
            # First pass: compute composite with default weights to get regime labels
            base_composite = sum(
                result[f"csi_{dim}"] * w for dim, w in weights.items()
                if f"csi_{dim}" in result
            ).clip(0, 100)
            regimes = regime_detector.predict(base_composite)
            result["csi_regime"] = regimes

            # Second pass: compute composite with per-day regime weights
            composite = pd.Series(0.0, index=result.index)
            for regime_id in range(3):
                mask = regimes == regime_id
                if not mask.any():
                    continue
                rw = get_regime_weights(regime_id)
                for dim, w in rw.items():
                    col = f"csi_{dim}"
                    if col in result:
                        composite[mask] += result.loc[mask, col] * w
            logger.info("CSI built with regime-conditioned weights.")
        except Exception as exc:
            logger.warning("Regime-conditioned weighting failed (%s); using defaults.", exc)
            composite = sum(
                result[f"csi_{dim}"] * w for dim, w in weights.items()
                if f"csi_{dim}" in result
            )
    else:
        composite = sum(
            result[f"csi_{dim}"] * w for dim, w in weights.items()
            if f"csi_{dim}" in result
        )
    result["csi_composite"] = composite.clip(0, 100)

    # Classification
    def classify(v):
        if pd.isna(v):
            return np.nan
        for label, (lo, hi) in CLASSIFICATION.items():
            if lo <= v <= hi:
                return label
        return "Extreme"

    result["csi_class"] = result["csi_composite"].apply(classify)

    if save:
        os.makedirs(labels_dir, exist_ok=True)
        result.to_parquet(os.path.join(labels_dir, "csi.parquet"))
        logger.info("Saved CSI to %s/csi.parquet", labels_dir)

    return result


def build_prediction_targets(
    prices: pd.DataFrame,
    csi: pd.DataFrame,
    horizons: list = None,
    labels_dir: str = "data/labels",
    save: bool = True,
) -> pd.DataFrame:
    """
    Build forward-looking prediction targets for each horizon.

    For each horizon h:
        stress_fwd_{h}d   : CSI composite h days ahead
        stress_delta_{h}d : change in CSI (stress direction magnitude)
        stress_dir_{h}d   : 1=increasing stress, -1=decreasing, 0=neutral
        market_ret_{h}d   : SPY log return over next h days
        market_dir_{h}d   : 1=Up, -1=Down, 0=Neutral (±1% threshold)
    """
    horizons = horizons or [5, 21, 63]
    targets = pd.DataFrame(index=csi.index)
    csi_comp = csi["csi_composite"]

    up_thresh = 0.01
    dn_thresh = -0.01

    if "SPY" in prices.columns:
        spy_log = np.log(prices["SPY"])
    else:
        spy_log = None

    for h in horizons:
        # Stress targets
        fwd_csi = csi_comp.shift(-h)
        targets[f"stress_fwd_{h}d"]   = fwd_csi
        targets[f"stress_delta_{h}d"] = fwd_csi - csi_comp
        delta = targets[f"stress_delta_{h}d"]
        targets[f"stress_dir_{h}d"] = np.where(delta > 5, 1, np.where(delta < -5, -1, 0))

        # Market targets (SPY)
        if spy_log is not None:
            fwd_ret = spy_log.shift(-h) - spy_log
            targets[f"market_ret_{h}d"] = fwd_ret
            targets[f"market_dir_{h}d"] = np.where(
                fwd_ret > up_thresh, 1,
                np.where(fwd_ret < dn_thresh, -1, 0)
            )

    if save:
        os.makedirs(labels_dir, exist_ok=True)
        targets.to_parquet(os.path.join(labels_dir, "targets.parquet"))
        logger.info("Saved targets to %s/targets.parquet", labels_dir)

    return targets


def load_csi(labels_dir: str = "data/labels") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(labels_dir, "csi.parquet"))


def load_targets(labels_dir: str = "data/labels") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(labels_dir, "targets.parquet"))
