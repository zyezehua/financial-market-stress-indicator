"""Credit stress features: HY/IG spreads, price-implied credit stress."""

import numpy as np
import pandas as pd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_credit_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs
    ------
    prices : DataFrame with HYG, LQD, AGG columns
    macro  : DataFrame with BAMLH0A0HYM2 (HY OAS) and BAMLC0A0CM (IG OAS)

    Returns
    -------
    DataFrame of credit stress features
    """
    feat = pd.DataFrame(index=prices.index)

    # ── FRED OAS spreads (primary credit stress signal) ────────────────────
    if "BAMLH0A0HYM2" in macro.columns:
        hy_oas = macro["BAMLH0A0HYM2"]
        feat["cr_hy_oas"] = hy_oas
        feat["cr_hy_oas_pct_rank_252d"] = _rolling_pct_rank(hy_oas, 252)
        feat["cr_hy_oas_z_63d"] = (hy_oas - hy_oas.rolling(63).mean()) / hy_oas.rolling(63).std()
        feat["cr_hy_oas_mom_5d"] = hy_oas.diff(5)
        feat["cr_hy_oas_mom_21d"] = hy_oas.diff(21)

    if "BAMLC0A0CM" in macro.columns:
        ig_oas = macro["BAMLC0A0CM"]
        feat["cr_ig_oas"] = ig_oas
        feat["cr_ig_oas_pct_rank_252d"] = _rolling_pct_rank(ig_oas, 252)
        feat["cr_ig_oas_z_63d"] = (ig_oas - ig_oas.rolling(63).mean()) / ig_oas.rolling(63).std()

    # ── AAA OAS (highest-quality credit spread) ───────────────────────────
    if "BAMLC0A1CAAA" in macro.columns:
        aaa_oas = macro["BAMLC0A1CAAA"]
        feat["cr_aaa_oas"] = aaa_oas
        feat["cr_aaa_oas_pct_rank_252d"] = _rolling_pct_rank(aaa_oas, 252)
        feat["cr_aaa_oas_mom_21d"] = aaa_oas.diff(21)

    # ── HY minus IG spread (pure credit risk premium) ─────────────────────
    if "BAMLH0A0HYM2" in macro.columns and "BAMLC0A0CM" in macro.columns:
        hy_ig_spread = macro["BAMLH0A0HYM2"] - macro["BAMLC0A0CM"]
        feat["cr_hy_ig_spread"] = hy_ig_spread
        feat["cr_hy_ig_spread_pct_rank_252d"] = _rolling_pct_rank(hy_ig_spread, 252)

    # ── IG minus AAA spread (BBB/single-A stress premium) ────────────────
    if "BAMLC0A0CM" in macro.columns and "BAMLC0A1CAAA" in macro.columns:
        ig_aaa_spread = macro["BAMLC0A0CM"] - macro["BAMLC0A1CAAA"]
        feat["cr_ig_aaa_spread"] = ig_aaa_spread
        feat["cr_ig_aaa_spread_pct_rank_252d"] = _rolling_pct_rank(ig_aaa_spread, 252)

    # ── ETF-based credit features (HYG, LQD) ──────────────────────────────
    if "HYG" in prices.columns:
        hyg_ret = prices["HYG"].pct_change()
        feat["cr_hyg_rvol_21d"] = _realized_vol(hyg_ret, 21)
        feat["cr_hyg_rvol_21d_pct_rank_252d"] = _rolling_pct_rank(feat["cr_hyg_rvol_21d"], 252)
        feat["cr_hyg_ret_5d"] = hyg_ret.rolling(5).sum()
        feat["cr_hyg_ret_21d"] = hyg_ret.rolling(21).sum()
        feat["cr_hyg_drawdown_63d"] = (
            prices["HYG"] / prices["HYG"].rolling(63).max() - 1
        ).clip(upper=0).abs()

    if "LQD" in prices.columns:
        lqd_ret = prices["LQD"].pct_change()
        feat["cr_lqd_rvol_21d"] = _realized_vol(lqd_ret, 21)
        feat["cr_lqd_ret_21d"] = lqd_ret.rolling(21).sum()

    # ── HYG / SPY relative return (credit-equity divergence) ──────────────
    if "HYG" in prices.columns and "SPY" in prices.columns:
        hyg_spy_rel = prices["HYG"] / prices["SPY"]
        hyg_spy_rel = hyg_spy_rel / hyg_spy_rel.rolling(252).mean()
        feat["cr_hyg_spy_rel_norm"] = hyg_spy_rel
        feat["cr_hyg_spy_rel_mom_21d"] = hyg_spy_rel.pct_change(21)

    # ── LQD / AGG spread (corporate vs aggregate bond premium) ────────────
    if "LQD" in prices.columns and "AGG" in prices.columns:
        lqd_agg_rel = prices["LQD"] / prices["AGG"]
        feat["cr_lqd_agg_rel_mom_21d"] = lqd_agg_rel.pct_change(21)

    # ── HYG / TLT ratio — credit stress decoupled from duration ──────────
    # When HYG falls while TLT rallies (ratio drops sharply), it signals pure
    # credit tightening + flight to safety simultaneously — a hallmark of
    # Extreme stress (GFC 2008, COVID 2020). Distinct from HYG/SPY which
    # captures credit-equity divergence only.
    if "HYG" in prices.columns and "TLT" in prices.columns:
        hyg_tlt = prices["HYG"] / prices["TLT"]
        hyg_tlt_ret = hyg_tlt.pct_change()
        feat["cr_hyg_tlt_ratio"] = hyg_tlt
        feat["cr_hyg_tlt_ratio_pct_rank_252d"] = _rolling_pct_rank(hyg_tlt, 252)
        feat["cr_hyg_tlt_ret_5d"]  = hyg_tlt_ret.rolling(5).sum()
        feat["cr_hyg_tlt_ret_21d"] = hyg_tlt_ret.rolling(21).sum()
        feat["cr_hyg_tlt_drawdown_63d"] = (
            hyg_tlt / hyg_tlt.rolling(63).max() - 1
        ).clip(upper=0).abs()

    # ── HYG 252d drawdown (longer-horizon credit deterioration) ──────────
    if "HYG" in prices.columns:
        feat["cr_hyg_drawdown_252d"] = (
            prices["HYG"] / prices["HYG"].rolling(252).max() - 1
        ).clip(upper=0).abs()

    return feat.ffill().bfill()
