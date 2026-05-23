"""
Human-readable factor attribution: maps raw feature names to readable labels
and builds structured explanations for each prediction.
"""

import pandas as pd

# Maps feature prefixes / exact names to readable descriptions
FEATURE_LABELS = {
    # Equity
    "eq_vix_level":                        "VIX level",
    "eq_vix_pct_rank_252d":                "VIX (1Y percentile rank)",
    "eq_vix_term_ratio":                   "VIX term structure ratio (VIX/VIX3M)",
    "eq_vix_term_ratio_pct_rank_252d":     "VIX backwardation (1Y pct rank)",
    "eq_vvix_level":                       "VVIX (vol-of-vol) level",
    "eq_vvix_pct_rank_252d":               "VVIX (1Y percentile rank)",
    "eq_spy_rvol_5d":                      "SPY 5-day realized vol",
    "eq_spy_rvol_21d":                     "SPY 21-day realized vol",
    "eq_spy_rvol_21d_pct_rank_252d":       "SPY realized vol (1Y pct rank)",
    "eq_spy_drawdown_252d":                "SPY drawdown from 1Y high",
    "eq_spy_ret_5d":                       "SPY 5-day return",
    "eq_spy_ret_21d":                      "SPY 21-day return",
    "eq_spy_tlt_corr_21d":                 "SPY-TLT 21-day correlation",
    "eq_qqq_spy_rel_mom_21d":              "QQQ/SPY relative momentum (21d)",
    # Credit
    "cr_hy_oas":                           "HY option-adjusted spread",
    "cr_hy_oas_pct_rank_252d":             "HY OAS (1Y percentile rank)",
    "cr_ig_oas":                           "IG option-adjusted spread",
    "cr_hy_ig_spread":                     "HY minus IG spread",
    "cr_hy_ig_spread_pct_rank_252d":       "HY-IG spread (1Y pct rank)",
    "cr_hyg_rvol_21d":                     "HYG 21-day realized vol",
    "cr_hyg_rvol_21d_pct_rank_252d":       "HYG realized vol (1Y pct rank)",
    "cr_hyg_drawdown_63d":                 "HYG drawdown from 3-month high",
    "cr_hyg_spy_rel_mom_21d":              "HYG/SPY relative momentum (credit-equity divergence)",
    # Rates
    "rt_t10y_level":                       "10-Year Treasury yield",
    "rt_t10y_change_5d":                   "10Y yield 5-day change",
    "rt_t10y_abs_change_5d_pct_rank_252d": "10Y yield rate-of-change (1Y pct rank)",
    "rt_t10y2y_spread":                    "10Y-2Y yield spread",
    "rt_inversion_depth":                  "Yield curve inversion depth",
    "rt_inversion_depth_pct_rank_252d":    "Yield curve inversion (1Y pct rank)",
    "rt_tlt_rvol_21d":                     "TLT 21-day realized vol (MOVE proxy)",
    "rt_tlt_rvol_21d_pct_rank_252d":       "Rates realized vol (1Y pct rank)",
    # Liquidity
    "lq_funding_spread":                   "SOFR-3M T-bill funding spread",
    "lq_funding_spread_pct_rank_252d":     "Funding stress (1Y pct rank)",
    "lq_avg_cross_corr_21d":              "Cross-asset avg correlation (21d)",
    "lq_avg_cross_corr_21d_pct_rank_252d": "Contagion indicator (1Y pct rank)",
    "lq_spy_tlt_corr_21d":                "SPY-TLT correlation (flight-to-quality)",
    # FX
    "fx_usd_pct_rank_252d":               "USD strength (1Y pct rank)",
    "fx_usdjpy_level":                    "USD/JPY level",
    "fx_jpy_strength_pct_rank_252d":      "JPY safe-haven demand (1Y pct rank)",
    "fx_eurusd_rvol_21d":                 "EUR/USD 21-day realized vol",
    "fx_vol_composite_pct_rank_252d":     "FX volatility composite (1Y pct rank)",
    # International
    "intl_eem_rvol_21d":                  "EM realized vol (21d)",
    "intl_eem_rvol_pct_rank_252d":        "EM volatility (1Y pct rank)",
    "intl_eem_spy_rel_mom_21d":           "EM/US relative performance (21d)",
    "intl_vol_composite_pct_rank_252d":   "Global vol composite (1Y pct rank)",
    "intl_gld_pct_rank_252d":             "Gold level (1Y pct rank)",
}

DIMENSION_MAP = {
    "eq_":   "Equity",
    "cr_":   "Credit",
    "rt_":   "Rates",
    "lq_":   "Liquidity",
    "fx_":   "FX",
    "intl_": "International",
}


def readable_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def get_dimension(feature: str) -> str:
    for prefix, dim in DIMENSION_MAP.items():
        if feature.startswith(prefix):
            return dim
    return "Other"


def build_attribution(
    top_shap: pd.Series,
    current_values: pd.Series,
    feature_percentiles: pd.Series = None,
) -> list:
    """
    Build a structured list of factor attributions for the top-N SHAP features.

    Each entry:
        feature, readable_name, dimension, shap_value, current_value, percentile
    """
    attribution = []
    for feat, shap_val in top_shap.items():
        current = float(current_values.get(feat, float("nan")))
        pct = float(feature_percentiles.get(feat, float("nan"))) if feature_percentiles is not None else float("nan")
        attribution.append({
            "feature":       feat,
            "label":         readable_name(feat),
            "dimension":     get_dimension(feat),
            "shap_value":    round(float(shap_val), 4),
            "current_value": round(current, 3) if not pd.isna(current) else None,
            "percentile":    round(pct, 1) if not pd.isna(pct) else None,
        })
    return attribution


def current_feature_percentiles(
    features: pd.DataFrame,
    row_index,
    window: int = 252,
) -> pd.Series:
    """Compute the rolling percentile rank of the current row vs trailing window."""
    end_loc = features.index.get_loc(row_index)
    start_loc = max(0, end_loc - window + 1)
    window_data = features.iloc[start_loc: end_loc + 1]
    current_row = window_data.iloc[-1]
    ranks = window_data.rank(pct=True).iloc[-1] * 100
    return ranks
