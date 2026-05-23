"""FX stress features: USD strength, safe-haven flows, currency volatility."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_fx_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with UUP (DXY proxy), USDJPY=X, EURUSD=X, GBPUSD=X
    macro  : DataFrame (unused in this module)

    Returns
    -------
    DataFrame of FX stress features
    """
    feat = pd.DataFrame(index=prices.index)

    # ── USD strength (DXY proxy via UUP ETF) ──────────────────────────────
    if "UUP" in prices.columns:
        uup = prices["UUP"]
        uup_ret = uup.pct_change()
        feat["fx_usd_level_norm"] = uup / uup.rolling(252).mean()
        feat["fx_usd_ret_5d"] = uup_ret.rolling(5).sum()
        feat["fx_usd_ret_21d"] = uup_ret.rolling(21).sum()
        feat["fx_usd_rvol_21d"] = _realized_vol(uup_ret, 21)
        feat["fx_usd_pct_rank_252d"] = _rolling_pct_rank(uup, 252)

    # ── USD/JPY — safe-haven signal (falling USDJPY = JPY strengthening = risk-off)
    if "USDJPY=X" in prices.columns:
        usdjpy = prices["USDJPY=X"]
        usdjpy_ret = usdjpy.pct_change()
        feat["fx_usdjpy_level"] = usdjpy
        feat["fx_usdjpy_ret_5d"] = usdjpy_ret.rolling(5).sum()
        feat["fx_usdjpy_ret_21d"] = usdjpy_ret.rolling(21).sum()
        feat["fx_usdjpy_rvol_21d"] = _realized_vol(usdjpy_ret, 21)
        # Invert: high stress = low USDJPY (JPY appreciation)
        usdjpy_inv = 1 / usdjpy
        feat["fx_jpy_strength_pct_rank_252d"] = _rolling_pct_rank(usdjpy_inv, 252)
        # 52-week high distance (JPY far from high = complacency)
        jpy_52wk_high = usdjpy_inv.rolling(252).max()
        feat["fx_jpy_distance_from_52wk_high"] = 1 - usdjpy_inv / jpy_52wk_high

    # ── EUR/USD volatility ─────────────────────────────────────────────────
    if "EURUSD=X" in prices.columns:
        eurusd = prices["EURUSD=X"]
        eurusd_ret = eurusd.pct_change()
        feat["fx_eurusd_ret_21d"] = eurusd_ret.rolling(21).sum()
        feat["fx_eurusd_rvol_21d"] = _realized_vol(eurusd_ret, 21)
        feat["fx_eurusd_rvol_pct_rank_252d"] = _rolling_pct_rank(
            feat["fx_eurusd_rvol_21d"], 252
        )

    # ── GBP/USD ───────────────────────────────────────────────────────────
    if "GBPUSD=X" in prices.columns:
        gbpusd = prices["GBPUSD=X"]
        gbpusd_ret = gbpusd.pct_change()
        feat["fx_gbpusd_rvol_21d"] = _realized_vol(gbpusd_ret, 21)

    # ── Multi-currency volatility composite ────────────────────────────────
    fx_vols = [feat[c] for c in ["fx_usdjpy_rvol_21d", "fx_eurusd_rvol_21d"] if c in feat]
    if len(fx_vols) >= 2:
        fx_vol_composite = sum(fx_vols) / len(fx_vols)
        feat["fx_vol_composite"] = fx_vol_composite
        feat["fx_vol_composite_pct_rank_252d"] = _rolling_pct_rank(fx_vol_composite, 252)

    return feat.ffill().bfill()
