"""Liquidity stress features: funding spreads, cross-asset correlations, contagion."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _avg_pairwise_corr(returns_df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling average pairwise correlation across assets (contagion indicator)."""
    n = len(returns_df.columns)
    if n < 2:
        return pd.Series(np.nan, index=returns_df.index)

    corr_sum = pd.Series(0.0, index=returns_df.index)
    count = 0
    cols = returns_df.columns.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum += (
                returns_df[cols[i]].rolling(window).corr(returns_df[cols[j]])
            )
            count += 1
    return corr_sum / count if count > 0 else corr_sum


def compute_liquidity_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with SPY, TLT, HYG, GLD, EEM (for cross-asset correlations)
    macro  : DataFrame with SOFR, DGS3MO (for funding spread)

    Returns
    -------
    DataFrame of liquidity stress features
    """
    feat = pd.DataFrame(index=prices.index)

    # ── SOFR-based funding spread ──────────────────────────────────────────
    # SOFR vs 3M T-bill spread captures interbank/repo funding stress
    if "SOFR" in macro.columns and "DGS3MO" in macro.columns:
        funding_spread = macro["SOFR"] - macro["DGS3MO"]
        feat["lq_funding_spread"] = funding_spread
        feat["lq_funding_spread_pct_rank_252d"] = _rolling_pct_rank(funding_spread, 252)
        feat["lq_funding_spread_z_63d"] = (
            (funding_spread - funding_spread.rolling(63).mean())
            / funding_spread.rolling(63).std()
        )
    elif "SOFR" in macro.columns:
        sofr = macro["SOFR"]
        feat["lq_sofr_level"] = sofr
        feat["lq_sofr_change_5d"] = sofr.diff(5)

    # ── SOFR-OIS spread (SOFR vs Fed Funds rate) ───────────────────────────
    # Secured vs unsecured overnight spread: pure credit-risk premium signal.
    # Spikes during interbank stress (GFC repo squeeze, SVB money-market flight).
    if "SOFR" in macro.columns and "DFF" in macro.columns:
        sofr_ois = macro["SOFR"] - macro["DFF"]
        feat["lq_sofr_ois_spread"] = sofr_ois
        feat["lq_sofr_ois_spread_z_63d"] = (
            (sofr_ois - sofr_ois.rolling(63).mean())
            / sofr_ois.rolling(63).std()
        )
        feat["lq_sofr_ois_spread_pct_rank_252d"] = _rolling_pct_rank(sofr_ois, 252)

    # ── Cross-asset average pairwise correlation (contagion) ───────────────
    # High correlation across uncorrelated assets signals systemic stress/flight
    cross_assets = [c for c in ["SPY", "TLT", "HYG", "GLD", "EEM"] if c in prices.columns]
    if len(cross_assets) >= 3:
        returns = prices[cross_assets].pct_change()
        avg_corr_21 = _avg_pairwise_corr(returns, 21)
        avg_corr_63 = _avg_pairwise_corr(returns, 63)
        feat["lq_avg_cross_corr_21d"] = avg_corr_21
        feat["lq_avg_cross_corr_63d"] = avg_corr_63
        feat["lq_avg_cross_corr_21d_pct_rank_252d"] = _rolling_pct_rank(avg_corr_21, 252)

    # ── SPY / TLT correlation (equity-bond relationship) ──────────────────
    if "SPY" in prices.columns and "TLT" in prices.columns:
        spy_r = prices["SPY"].pct_change()
        tlt_r = prices["TLT"].pct_change()
        corr = spy_r.rolling(21).corr(tlt_r)
        feat["lq_spy_tlt_corr_21d"] = corr
        feat["lq_spy_tlt_corr_21d_pct_rank_252d"] = _rolling_pct_rank(corr, 252)

    # ── SPY / GLD correlation (risk-off / safe-haven demand) ──────────────
    if "SPY" in prices.columns and "GLD" in prices.columns:
        spy_r = prices["SPY"].pct_change()
        gld_r = prices["GLD"].pct_change()
        feat["lq_spy_gld_corr_21d"] = spy_r.rolling(21).corr(gld_r)

    # ── HYG / TLT return dispersion (credit-rates divergence) ─────────────
    if "HYG" in prices.columns and "TLT" in prices.columns:
        hyg_r = prices["HYG"].pct_change()
        tlt_r = prices["TLT"].pct_change()
        feat["lq_hyg_tlt_corr_21d"] = hyg_r.rolling(21).corr(tlt_r)

    # ── Volume stress proxy (SPY volume spike) ────────────────────────────
    # High volume relative to 21d average often coincides with stress episodes
    if "SPY" in prices.columns:
        # We don't have volume in "Adj Close" fetch; skip unless OHLCV is available
        pass

    return feat.ffill().bfill()
