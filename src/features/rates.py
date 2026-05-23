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
        # Realized rate volatility — rolling std of daily yield changes (bps).
        # Direct MOVE-index analogue derived from rate history; captures taper
        # tantrum (2013), 2022 bear market, 2024 repricing without needing
        # swaption data. Distinct from TLT price vol (no duration/convexity noise).
        t10_daily_chg = t10.diff() * 100          # pct → bps
        rt_rvol_21 = t10_daily_chg.rolling(21).std()
        feat["rt_t10y_rvol_21d_bp"] = rt_rvol_21
        feat["rt_t10y_rvol_21d_bp_pct_rank_252d"] = _rolling_pct_rank(rt_rvol_21, 252)
        feat["rt_t10y_rvol_63d_bp"] = t10_daily_chg.rolling(63).std()

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

    # ── 5-Year and 30-Year yields ─────────────────────────────────────────
    if "DGS5" in macro.columns:
        t5 = macro["DGS5"]
        feat["rt_t5y_level"] = t5
        feat["rt_t5y_change_5d"] = t5.diff(5)

    if "DGS30" in macro.columns:
        t30 = macro["DGS30"]
        feat["rt_t30y_level"] = t30
        feat["rt_t30y_change_5d"] = t30.diff(5)

    # ── Butterfly spread (5Y - 0.5*(2Y+10Y)) — curve concavity ──────────
    # Negative butterfly = humped curve = anticipation of policy reversal
    if all(c in macro.columns for c in ("DGS2", "DGS5", "DGS10")):
        butterfly = macro["DGS5"] - 0.5 * (macro["DGS2"] + macro["DGS10"])
        feat["rt_butterfly_2_5_10"] = butterfly
        feat["rt_butterfly_pct_rank_252d"] = _rolling_pct_rank(butterfly, 252)

    # ── 30Y-10Y term premium ──────────────────────────────────────────────
    if "DGS10" in macro.columns and "DGS30" in macro.columns:
        term_prem = macro["DGS30"] - macro["DGS10"]
        feat["rt_t30y_t10y_spread"] = term_prem
        feat["rt_t30y_t10y_pct_rank_252d"] = _rolling_pct_rank(term_prem, 252)

    # ── 10Y TIPS real yield ───────────────────────────────────────────────
    # Rising real yields tighten financial conditions → stress signal
    if "DFII10" in macro.columns:
        real_yield = macro["DFII10"]
        feat["rt_tips10y_level"] = real_yield
        feat["rt_tips10y_pct_rank_252d"] = _rolling_pct_rank(real_yield, 252)
        feat["rt_tips10y_change_21d"] = real_yield.diff(21)
        # Breakeven inflation = nominal 10Y - real 10Y
        if "DGS10" in macro.columns:
            breakeven = macro["DGS10"] - real_yield
            feat["rt_breakeven_inflation"] = breakeven
            feat["rt_breakeven_pct_rank_252d"] = _rolling_pct_rank(breakeven, 252)

    # ── Mortgage spread (30Y mortgage - 10Y Treasury) ─────────────────────
    # Widening mortgage spread signals housing/consumer financial stress
    if "MORTGAGE30US" in macro.columns and "DGS10" in macro.columns:
        mortgage_spread = macro["MORTGAGE30US"] - macro["DGS10"]
        feat["rt_mortgage_spread"] = mortgage_spread
        feat["rt_mortgage_spread_pct_rank_252d"] = _rolling_pct_rank(mortgage_spread, 252)
        feat["rt_mortgage_spread_change_21d"] = mortgage_spread.diff(21)

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
