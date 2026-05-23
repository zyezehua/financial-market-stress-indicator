"""Rates stress features: yield curve shape, MOVE proxy, rate velocity."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_rates_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with TLT column (used as MOVE proxy via realized vol)
    macro  : DataFrame with DGS2, DGS10, DGS3MO, T10Y2Y

    Returns
    -------
    DataFrame of rates stress features
    """
    feat = pd.DataFrame(index=prices.index)

    # ── 10-Year yield level and velocity ──────────────────────────────────
    if "DGS10" in macro.columns:
        t10 = macro["DGS10"]
        feat["rt_t10y_level"] = t10
        feat["rt_t10y_pct_rank_252d"] = _rolling_pct_rank(t10, 252)
        feat["rt_t10y_change_5d"] = t10.diff(5)
        feat["rt_t10y_change_21d"] = t10.diff(21)
        feat["rt_t10y_abs_change_5d"] = t10.diff(5).abs()
        feat["rt_t10y_abs_change_5d_pct_rank_252d"] = _rolling_pct_rank(
            feat["rt_t10y_abs_change_5d"], 252
        )

    # ── 2-Year yield level ────────────────────────────────────────────────
    if "DGS2" in macro.columns:
        t2 = macro["DGS2"]
        feat["rt_t2y_level"] = t2
        feat["rt_t2y_change_5d"] = t2.diff(5)

    # ── Yield curve shape (2s10s) ─────────────────────────────────────────
    # Deep inversion (negative spread) signals elevated recession/stress risk
    if "T10Y2Y" in macro.columns:
        spread = macro["T10Y2Y"]
        feat["rt_t10y2y_spread"] = spread
        feat["rt_t10y2y_pct_rank_252d"] = _rolling_pct_rank(spread, 252)
        # Inversion depth: 0 when positive, increases as curve inverts deeper
        feat["rt_inversion_depth"] = (-spread).clip(lower=0)
        feat["rt_inversion_depth_pct_rank_252d"] = _rolling_pct_rank(
            feat["rt_inversion_depth"], 252
        )
    elif "DGS10" in macro.columns and "DGS2" in macro.columns:
        spread = macro["DGS10"] - macro["DGS2"]
        feat["rt_t10y2y_spread"] = spread
        feat["rt_inversion_depth"] = (-spread).clip(lower=0)

    # ── 3M yield (front end) ──────────────────────────────────────────────
    if "DGS3MO" in macro.columns:
        t3mo = macro["DGS3MO"]
        feat["rt_t3mo_level"] = t3mo
        if "DGS10" in macro.columns:
            feat["rt_t10y_3mo_spread"] = macro["DGS10"] - t3mo

    # ── TLT as MOVE proxy (bond market realized vol) ──────────────────────
    if "TLT" in prices.columns:
        tlt_ret = prices["TLT"].pct_change()
        tlt_rvol_21 = _realized_vol(tlt_ret, 21)
        feat["rt_tlt_rvol_21d"] = tlt_rvol_21
        feat["rt_tlt_rvol_21d_pct_rank_252d"] = _rolling_pct_rank(tlt_rvol_21, 252)
        feat["rt_tlt_rvol_63d"] = _realized_vol(tlt_ret, 63)
        feat["rt_tlt_ret_21d"] = tlt_ret.rolling(21).sum()
        feat["rt_tlt_drawdown_63d"] = (
            prices["TLT"] / prices["TLT"].rolling(63).max() - 1
        ).clip(upper=0).abs()

    return feat.ffill().bfill()
