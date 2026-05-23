"""International / auxiliary features: EM stress, Europe, Japan relative performance."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_international_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with EEM, VGK, EWJ, SPY columns
    macro  : DataFrame (unused in this module)

    Returns
    -------
    DataFrame of international stress features
    """
    feat = pd.DataFrame(index=prices.index)

    # ── EEM (Emerging Markets) ─────────────────────────────────────────────
    if "EEM" in prices.columns:
        eem = prices["EEM"]
        eem_ret = eem.pct_change()
        feat["intl_eem_rvol_21d"] = _realized_vol(eem_ret, 21)
        feat["intl_eem_rvol_pct_rank_252d"] = _rolling_pct_rank(feat["intl_eem_rvol_21d"], 252)
        feat["intl_eem_ret_5d"] = eem_ret.rolling(5).sum()
        feat["intl_eem_ret_21d"] = eem_ret.rolling(21).sum()
        feat["intl_eem_drawdown_63d"] = (
            eem / eem.rolling(63).max() - 1
        ).clip(upper=0).abs()

        # EEM / SPY relative strength (EM underperformance = global stress)
        if "SPY" in prices.columns:
            eem_spy = eem / prices["SPY"]
            feat["intl_eem_spy_rel_mom_21d"] = eem_spy.pct_change(21)
            feat["intl_eem_spy_rel_pct_rank_252d"] = _rolling_pct_rank(eem_spy, 252)

    # ── VGK (Europe ETF) ──────────────────────────────────────────────────
    if "VGK" in prices.columns:
        vgk = prices["VGK"]
        vgk_ret = vgk.pct_change()
        feat["intl_vgk_rvol_21d"] = _realized_vol(vgk_ret, 21)
        feat["intl_vgk_ret_21d"] = vgk_ret.rolling(21).sum()

        if "SPY" in prices.columns:
            vgk_spy = vgk / prices["SPY"]
            feat["intl_vgk_spy_rel_mom_21d"] = vgk_spy.pct_change(21)

    # ── EWJ (Japan ETF) ───────────────────────────────────────────────────
    if "EWJ" in prices.columns:
        ewj = prices["EWJ"]
        ewj_ret = ewj.pct_change()
        feat["intl_ewj_rvol_21d"] = _realized_vol(ewj_ret, 21)
        feat["intl_ewj_ret_21d"] = ewj_ret.rolling(21).sum()

    # ── Cross-regional volatility composite ────────────────────────────────
    intl_vols = [
        feat[c]
        for c in ["intl_eem_rvol_21d", "intl_vgk_rvol_21d", "intl_ewj_rvol_21d"]
        if c in feat
    ]
    if intl_vols:
        intl_vol_composite = sum(intl_vols) / len(intl_vols)
        feat["intl_vol_composite"] = intl_vol_composite
        feat["intl_vol_composite_pct_rank_252d"] = _rolling_pct_rank(intl_vol_composite, 252)

    # ── GLD (Gold) — global safe-haven demand ─────────────────────────────
    if "GLD" in prices.columns:
        gld = prices["GLD"]
        gld_ret = gld.pct_change()
        feat["intl_gld_ret_5d"] = gld_ret.rolling(5).sum()
        feat["intl_gld_ret_21d"] = gld_ret.rolling(21).sum()
        feat["intl_gld_pct_rank_252d"] = _rolling_pct_rank(gld, 252)

    return feat.ffill().bfill()
