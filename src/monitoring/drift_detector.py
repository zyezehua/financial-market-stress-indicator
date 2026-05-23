"""
Feature drift detector using Population Stability Index (PSI).

PSI measures how much a feature's distribution has shifted between a reference
period (training window) and a monitoring period (recent N days).

PSI interpretation
──────────────────
  < 0.10  : No significant change — model stable
  0.10–0.20: Moderate change — monitor closely
  > 0.20  : Significant drift — consider retraining

Reference: Siddiqi (2006), Credit Risk Scorecards.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_STABLE    = 0.10
PSI_MODERATE  = 0.20
N_BINS        = 10      # buckets for PSI histogram


def _psi_series(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Compute PSI between reference and current distributions.

    Both arrays should be 1-D, same feature. NaNs are dropped before computation.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) < 10 or len(cur) < 10:
        return np.nan

    # Build bins from reference distribution
    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6   # ensure lowest value is captured
    bins[-1] += 1e-6

    ref_counts = np.histogram(ref, bins=bins)[0]
    cur_counts = np.histogram(cur, bins=bins)[0]

    # Avoid zero counts (replace with small epsilon)
    eps = 0.5
    ref_pct = (ref_counts + eps) / (len(ref) + eps * n_bins)
    cur_pct = (cur_counts + eps) / (len(cur) + eps * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi(
    features: pd.DataFrame,
    reference_end: Optional[str] = None,
    monitor_window: int = 63,
    reference_window: int = 756,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """
    Compute per-feature PSI between a reference period and a recent window.

    Parameters
    ----------
    features         : full feature matrix
    reference_end    : last date of the reference period (default: 756d before end of data)
    monitor_window   : number of recent trading days to treat as "current" distribution
    reference_window : number of days before reference_end to use as reference
    n_bins           : histogram bins for PSI

    Returns
    -------
    DataFrame with columns: feature, psi, status ('stable'/'moderate'/'drift')
    sorted descending by PSI.
    """
    idx = features.index

    if reference_end is None:
        ref_end_pos = len(idx) - monitor_window
    else:
        ref_end_pos = int(idx.searchsorted(pd.Timestamp(reference_end)))

    ref_start_pos = max(0, ref_end_pos - reference_window)
    mon_start_pos = max(0, len(idx) - monitor_window)

    reference_df = features.iloc[ref_start_pos:ref_end_pos]
    current_df   = features.iloc[mon_start_pos:]

    records = []
    for col in features.columns:
        psi_val = _psi_series(
            reference_df[col].values, current_df[col].values, n_bins=n_bins
        )
        if np.isnan(psi_val):
            continue
        if psi_val < PSI_STABLE:
            status = "stable"
        elif psi_val < PSI_MODERATE:
            status = "moderate"
        else:
            status = "drift"
        records.append({"feature": col, "psi": round(psi_val, 4), "status": status})

    result = (
        pd.DataFrame(records)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
    return result


def summarize_drift(psi_df: pd.DataFrame) -> dict:
    """
    Return a high-level drift summary dict.

    Keys: n_stable, n_moderate, n_drift, top_drifting (list of feature names),
          retrain_recommended (bool)
    """
    counts = psi_df["status"].value_counts().to_dict()
    top = psi_df[psi_df["status"] == "drift"]["feature"].tolist()[:10]
    return {
        "n_stable":            counts.get("stable",   0),
        "n_moderate":          counts.get("moderate", 0),
        "n_drift":             counts.get("drift",    0),
        "top_drifting":        top,
        "retrain_recommended": counts.get("drift", 0) > 5,
    }


def log_drift_report(psi_df: pd.DataFrame) -> None:
    summary = summarize_drift(psi_df)
    logger.info(
        "PSI drift check | stable=%d | moderate=%d | drift=%d | retrain=%s",
        summary["n_stable"], summary["n_moderate"], summary["n_drift"],
        summary["retrain_recommended"],
    )
    if summary["top_drifting"]:
        logger.warning("Top drifting features: %s", summary["top_drifting"][:5])
    if summary["retrain_recommended"]:
        logger.warning(
            "DRIFT ALERT: %d features show significant distribution shift. "
            "Consider retraining models.",
            summary["n_drift"],
        )
