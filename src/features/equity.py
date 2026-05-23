"""Equity stress features: VIX structure, realized vol, momentum, breadth proxies."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value within trailing window (0-100)."""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def compute_equity_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with columns including ^VIX, ^VIX3M, ^VVIX, SPY, QQQ
    macro  : DataFrame (unused in this module, kept for API consistency)

    Returns
    -------
    DataFrame of equity stress features aligned to prices.index
    """
    feat = pd.DataFrame(index=prices.index)

    # ── VIX level ──────────────────────────────────────────────────────────
    if "^VIX" in prices.columns:
        vix = prices["^VIX"]
        feat["eq_vix_level"] = vix
        feat["eq_vix_pct_rank_252d"] = _rolling_pct_rank(vix, 252)
        feat["eq_vix_z_21d"] = (vix - vix.rolling(21).mean()) / vix.rolling(21).std()
        feat["eq_vix_z_63d"] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
        feat["eq_vix_mom_5d"] = vix.pct_change(5)
        feat["eq_vix_mom_21d"] = vix.pct_change(21)

    # ── VIX term structure (VIX / VIX3M) ──────────────────────────────────
    # Ratio > 1 (backwardation) signals acute short-term stress
    if "^VIX" in prices.columns and "^VIX3M" in prices.columns:
        vix3m = prices["^VIX3M"].copy()
        # Before VIX3M data exists (~2011), proxy with VIX * 1.05 (typical spread)
        vix3m = vix3m.fillna(prices["^VIX"] * 1.05)
        ts_ratio = prices["^VIX"] / vix3m
        feat["eq_vix_term_ratio"] = ts_ratio
        feat["eq_vix_term_ratio_pct_rank_252d"] = _rolling_pct_rank(ts_ratio, 252)

    # ── VVIX (vol of vol) ──────────────────────────────────────────────────
    if "^VVIX" in prices.columns:
        vvix = prices["^VVIX"].copy()
        vvix = vvix.ffill()
        feat["eq_vvix_level"] = vvix
        feat["eq_vvix_pct_rank_252d"] = _rolling_pct_rank(vvix, 252)

    # ── SPY realized volatility ────────────────────────────────────────────
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change()
        feat["eq_spy_rvol_5d"] = _realized_vol(spy_ret, 5)
        feat["eq_spy_rvol_21d"] = _realized_vol(spy_ret, 21)
        feat["eq_spy_rvol_63d"] = _realized_vol(spy_ret, 63)
        feat["eq_spy_rvol_21d_pct_rank_252d"] = _rolling_pct_rank(feat["eq_spy_rvol_21d"], 252)

        feat["eq_spy_ret_5d"] = spy_ret.rolling(5).sum()
        feat["eq_spy_ret_21d"] = spy_ret.rolling(21).sum()
        feat["eq_spy_ret_63d"] = spy_ret.rolling(63).sum()

        # Drawdown from 252d high (measures how far we are from peak)
        high_252 = prices["SPY"].rolling(252).max()
        feat["eq_spy_drawdown_252d"] = (prices["SPY"] / high_252 - 1).clip(upper=0).abs()

    # ── SPY / QQQ relative strength (risk appetite) ────────────────────────
    if "SPY" in prices.columns and "QQQ" in prices.columns:
        qqq_spy_rel = prices["QQQ"] / prices["SPY"]
        feat["eq_qqq_spy_rel_mom_21d"] = qqq_spy_rel.pct_change(21)

    # ── SPY / TLT rolling correlation (flight-to-quality signal) ──────────
    if "SPY" in prices.columns and "TLT" in prices.columns:
        spy_r = prices["SPY"].pct_change()
        tlt_r = prices["TLT"].pct_change()
        corr_21 = spy_r.rolling(21).corr(tlt_r)
        feat["eq_spy_tlt_corr_21d"] = corr_21
        # Positive correlation (both falling) = systemic stress
        feat["eq_spy_tlt_corr_positive"] = (corr_21 > 0).astype(float)

    return feat.ffill().bfill()
