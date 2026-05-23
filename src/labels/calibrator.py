"""
CSI Calibrator: validates our Composite Stress Index against the FRED FSI benchmark.

Computes correlation, rank-correlation, and event-detection accuracy across
major historical stress episodes (dot-com, GFC, COVID, etc.).
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

STRESS_EPISODES = {
    "Dot-com Crash":      ("2000-03-01", "2002-10-31"),
    "9/11 Shock":         ("2001-09-01", "2001-10-31"),
    "GFC Peak":           ("2008-09-01", "2009-03-31"),
    "Euro Crisis":        ("2011-07-01", "2011-10-31"),
    "China Selloff":      ("2015-08-01", "2015-09-30"),
    "Covid Crash":        ("2020-02-20", "2020-04-30"),
    "2022 Bear Market":   ("2022-01-01", "2022-12-31"),
}


def compute_alignment(
    csi_composite: pd.Series,
    fred_fsi: pd.Series,
) -> dict:
    """
    Compute alignment statistics between CSI and FRED FSI.

    Returns a dict with:
        pearson_r, spearman_r, episode_recall
    """
    aligned = pd.concat(
        [csi_composite.rename("csi"), fred_fsi.rename("fsi")], axis=1
    ).dropna()

    if len(aligned) < 50:
        logger.warning("Too few overlapping observations for alignment check.")
        return {}

    pearson_r, pearson_p = stats.pearsonr(aligned["csi"], aligned["fsi"])
    spearman_r, spearman_p = stats.spearmanr(aligned["csi"], aligned["fsi"])

    # Episode detection: during each known stress episode, is CSI > 50?
    episode_hits = []
    for name, (start, end) in STRESS_EPISODES.items():
        ep = aligned.loc[start:end, "csi"]
        if ep.empty:
            continue
        pct_elevated = (ep > 50).mean()
        episode_hits.append(pct_elevated)
        logger.info("Episode %-22s → CSI > 50 on %.0f%% of days", name, pct_elevated * 100)

    result = {
        "pearson_r":     round(pearson_r, 3),
        "pearson_p":     round(pearson_p, 4),
        "spearman_r":    round(spearman_r, 3),
        "episode_recall": round(float(np.mean(episode_hits)), 3) if episode_hits else None,
        "n_obs":         len(aligned),
    }
    logger.info("CSI vs FRED FSI — Pearson: %.3f, Spearman: %.3f", pearson_r, spearman_r)
    return result


def adjust_csi_weights(
    features: pd.DataFrame,
    fred_fsi: pd.Series,
    initial_weights: dict,
    n_top: int = 3,
) -> dict:
    """
    Lightweight weight refinement: rank dimension scores by correlation with FRED FSI
    and nudge weights toward better-correlated dimensions.

    This is intentionally simple — a full optimization is overkill for Phase 1.
    """
    from .composite_index import _dimension_score

    corrs = {}
    for dim in initial_weights:
        score = _dimension_score(features, dim)
        aligned = pd.concat(
            [score.rename("score"), fred_fsi.rename("fsi")], axis=1
        ).dropna()
        if len(aligned) < 50:
            corrs[dim] = 0.0
            continue
        corrs[dim] = abs(stats.spearmanr(aligned["score"], aligned["fsi"])[0])

    logger.info("Dimension–FSI Spearman correlations: %s", corrs)

    total_corr = sum(corrs.values()) or 1.0
    adjusted = {dim: corrs[dim] / total_corr for dim in initial_weights}

    return adjusted
