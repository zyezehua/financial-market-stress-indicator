"""
Market Crisis Signal (MCS) — Entry and Exit lifecycle framework.

Evaluates model predictions against a realistic crisis detection standard:
  - ENTRY: did the model warn before or react early to a stress episode?
  - EXIT:  did the model identify when stress was normalizing?

Entry Conditions (any one = hit):
    C1  Persistent Buildup    rolling 21d mean >= 40, sustained >= 15 days, upward slope
    C2  Pre-Crisis Spike      crosses trailing 252d 85th-pct AND holds >= 3 days
    C3  Rapid Escalation      CSI rises >= 12 points in any 5-day window
    C4  Early Response        crosses >= 50 within first 10 trading days of T

Exit Conditions (any one = candidate; requires 15-day confirmation):
    E1  Sustained Normalization  rolling 21d mean < 38, holds >= 10 days
    E2  Rapid De-escalation      drops >= 15 pts in any 10-day window AND lands < 45
    E3  Level Cross-down         crosses < 50 AND 10d rolling mean also < 50
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Entry thresholds ──────────────────────────────────────────────────────────
C1_MEAN_THRESH  = 45    # rolling 21d mean threshold (raised from 40 to cut false alarms)
C1_STREAK       = 15    # consecutive days required
C2_PERCENTILE   = 90    # trailing 252d percentile for spike detection (raised from 85)
C2_HOLD         = 3     # days spike must hold
C3_RISE         = 12    # minimum point rise in window
C3_WIN          = 5     # window size in trading days
C4_THRESH       = 50    # cross-above threshold
C4_DAYS         = 10    # trading days after T_start

# ── Exit thresholds ───────────────────────────────────────────────────────────
ELEVATED_CONFIRM = 5    # consecutive days >= 50 needed to confirm elevated state
E1_MEAN_THRESH  = 35    # rolling 21d mean threshold for normalization (relaxed from 38)
E1_STREAK       = 7     # consecutive days for E1 (relaxed from 10)
E2_DROP         = 12    # minimum point drop in window (relaxed from 15)
E2_WIN          = 10    # window size in trading days
E2_LANDING      = 50    # must land below this (relaxed from 45)
E3_THRESH       = 50    # cross-below threshold
E3_MA           = 10    # rolling mean days for E3 confirmation
E3_STREAK       = 3     # consecutive days E3 condition must hold (new)
CONFIRM_WIN     = 15    # exit confirmation window in trading days
MIN_HOLD_DAYS   = 21    # minimum trading days after entry before exit scanning begins

# ── Detection windows ─────────────────────────────────────────────────────────
PRE_WIN         = 90    # Window A: trading days before T_start
ONSET_WIN       = 21    # Window B: trading days after T_start

# ── Regime-conditioned entry thresholds (keyed by HMM regime 0/1/2) ───────────
# Regime 0 (Low Stress):    lower bar — early buildup stands out vs calm baseline
# Regime 1 (Moderate):      default values from Phase A
# Regime 2 (High/Crisis):   stricter — baseline is already elevated, need genuine new spike
REGIME_ENTRY = {
    0: {"c1_thresh": 42, "c2_pct": 88},
    1: {"c1_thresh": 45, "c2_pct": 90},
    2: {"c1_thresh": 50, "c2_pct": 93},
}

# ── Regime-conditioned exit thresholds ────────────────────────────────────────
# Regime 0: exit confirmed sooner (calm environment recovers faster)
# Regime 2: require more sustained normalization before exiting
REGIME_EXIT = {
    0: {"e1_thresh": 32, "e1_streak": 5,  "min_hold": 15},
    1: {"e1_thresh": 35, "e1_streak": 7,  "min_hold": 21},
    2: {"e1_thresh": 38, "e1_streak": 10, "min_hold": 30},
}

# ── Crisis severity tiering ────────────────────────────────────────────────────
# Acute (peak CSI < 65):    isolated shock, exits on standard E3 (3-day streak)
# Structural (peak >= 65):  regime shift — require longer E3 + extra hold to avoid
#                           confirming exit during bear-market relief rallies
ACUTE_PEAK_THRESH    = 65
STRUCTURAL_MIN_HOLD  = 42   # trading days
STRUCTURAL_E3_STREAK = 10   # days (vs Acute's 3)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_streak(s: pd.Series, min_len: int) -> Optional[pd.Timestamp]:
    """
    Return the first date at which a run of True values of length >= min_len
    is CONFIRMED (i.e., the date the streak reaches min_len).
    Returns None if no such streak exists.
    """
    streak = 0
    for dt, val in s.items():
        if val:
            streak += 1
            if streak >= min_len:
                return dt
        else:
            streak = 0
    return None


def _streak_start(s: pd.Series, confirmed_date: pd.Timestamp, min_len: int) -> pd.Timestamp:
    """Given the confirmed date of a streak, return the date it started."""
    idx = s.index.get_loc(confirmed_date)
    start_idx = max(0, idx - min_len + 1)
    return s.index[start_idx]


def _any_rise(s: pd.Series, rise: float, window: int) -> Optional[pd.Timestamp]:
    """Return first date where s rose >= `rise` points over the previous `window` days."""
    diff = s - s.shift(window)
    hits = diff[diff >= rise]
    return hits.index[0] if len(hits) > 0 else None


def _any_drop(s: pd.Series, drop: float, window: int, landing: float) -> Optional[pd.Timestamp]:
    """Return first date where s dropped >= `drop` over `window` days AND value < landing."""
    diff = s.shift(window) - s     # positive = drop
    mask = (diff >= drop) & (s < landing)
    hits = s[mask]
    return hits.index[0] if len(hits) > 0 else None


def _slice(series: pd.Series, start, end) -> pd.Series:
    """Safe date-range slice (inclusive) that won't raise on missing dates."""
    return series.loc[
        series.index[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Entry signal detection
# ─────────────────────────────────────────────────────────────────────────────

def _check_c1(win_a: pd.Series, c1_thresh: float = C1_MEAN_THRESH) -> Optional[pd.Timestamp]:
    """C1: rolling 21d mean >= c1_thresh for 15 consecutive days, upward slope."""
    if len(win_a) < C1_STREAK:
        return None
    mean_21 = win_a.rolling(21, min_periods=10).mean()
    confirmed = _first_streak(mean_21 >= c1_thresh, C1_STREAK)
    if confirmed is None:
        return None
    start = _streak_start(mean_21 >= c1_thresh, confirmed, C1_STREAK)
    if mean_21[confirmed] > mean_21[start]:
        return start
    return None


def _check_c2(pred_full: pd.Series, win_a: pd.Series, c2_pct: float = C2_PERCENTILE) -> Optional[pd.Timestamp]:
    """C2: crosses trailing 252d c2_pct-percentile AND holds >= 3 days."""
    if len(win_a) < C2_HOLD:
        return None
    pct_val = pred_full.rolling(252, min_periods=126).quantile(c2_pct / 100)
    above = win_a > pct_val.reindex(win_a.index)
    confirmed = _first_streak(above, C2_HOLD)
    if confirmed is None:
        return None
    return _streak_start(above, confirmed, C2_HOLD)


def _check_c3(win_ab: pd.Series) -> Optional[pd.Timestamp]:
    """C3: CSI rises >= 12 points in any 5-day window."""
    return _any_rise(win_ab, C3_RISE, C3_WIN)


def _check_c4(win_b: pd.Series) -> Optional[pd.Timestamp]:
    """C4: crosses >= 50 within first 10 trading days of T_start."""
    first_10 = win_b.iloc[:C4_DAYS]
    hits = first_10[first_10 >= C4_THRESH]
    return hits.index[0] if len(hits) > 0 else None


def detect_entry(
    pred_full: pd.Series,
    T_start: str,
    regime: int = 1,
) -> dict:
    """
    Run all entry conditions for a single episode.

    Parameters
    ----------
    pred_full : full prediction series (all available history)
    T_start   : episode start date string (YYYY-MM-DD)
    regime    : HMM regime at T_start (0=Low, 1=Moderate, 2=High/Crisis)

    Returns
    -------
    dict with keys: hit, conditions_fired, signal_date, lead_days, grade
    """
    T = pd.Timestamp(T_start)
    entry_cfg = REGIME_ENTRY.get(regime, REGIME_ENTRY[1])

    win_a_start = pred_full.index[max(0, pred_full.index.searchsorted(T) - PRE_WIN)]
    win_a_end_pos = pred_full.index.searchsorted(T)
    if win_a_end_pos == 0:
        win_a = pd.Series(dtype=float)
    else:
        win_a = pred_full.iloc[max(0, win_a_end_pos - PRE_WIN): win_a_end_pos]

    win_b_end_pos = min(len(pred_full), pred_full.index.searchsorted(T) + ONSET_WIN)
    win_b = pred_full.iloc[pred_full.index.searchsorted(T): win_b_end_pos]

    win_ab_start = max(0, pred_full.index.searchsorted(T) - PRE_WIN)
    win_ab = pred_full.iloc[win_ab_start: win_b_end_pos]

    c1 = _check_c1(win_a, c1_thresh=entry_cfg["c1_thresh"])
    c2 = _check_c2(pred_full, win_a, c2_pct=entry_cfg["c2_pct"])
    c3 = _check_c3(win_ab)
    c4 = _check_c4(win_b)

    conditions = {}
    if c1 is not None: conditions["C1"] = c1
    if c2 is not None: conditions["C2"] = c2
    if c3 is not None: conditions["C3"] = c3
    if c4 is not None: conditions["C4"] = c4

    if not conditions:
        return {"hit": False, "conditions_fired": [], "signal_date": None,
                "lead_days": None, "grade": "✗ Miss"}

    # Pick earliest signal in Window A (best lead time), else C4
    window_a_hits = {k: v for k, v in conditions.items() if k != "C4"}
    if window_a_hits:
        earliest_key = min(window_a_hits, key=lambda k: window_a_hits[k])
        signal_date = window_a_hits[earliest_key]
        lead_days = int((T - signal_date).days)
        if lead_days >= 30:
            grade = "⭐⭐⭐ Early Warning"
        elif lead_days >= 10:
            grade = "⭐⭐ Timely Warning"
        else:
            grade = "⭐ Pre-Onset"
    else:
        signal_date = c4
        lead_days = -int((c4 - T).days)   # negative = after T
        grade = "✓ Rapid Detection"

    return {
        "hit":              True,
        "conditions_fired": list(conditions.keys()),
        "signal_date":      signal_date,
        "lead_days":        lead_days,
        "grade":            grade,
        "peak_in_win_a":    float(win_a.max()) if len(win_a) > 0 else np.nan,
        "days_above_40_a":  int((win_a >= 40).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exit signal detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_exit(
    pred_full: pd.Series,
    entry_date: pd.Timestamp,
    T_end_actual: str,
    regime: int = 1,
    severity: str = "Acute",
) -> dict:
    """
    Monitor for exit signal after entry has fired.

    Starts monitoring from entry_date. Requires:
    1. Model first reaches >= 50 for ELEVATED_CONFIRM consecutive days.
    2. Then monitors for E1 / E2 / E3 exit candidates.
    3. Each candidate must hold for CONFIRM_WIN trading days.

    Parameters
    ----------
    regime   : HMM regime (0=Low, 1=Moderate, 2=High/Crisis) — conditions exit thresholds
    severity : "Acute" or "Structural" — Structural requires longer hold + stricter E3
    """
    T_end = pd.Timestamp(T_end_actual)
    series = pred_full.loc[entry_date:] if entry_date in pred_full.index else pred_full

    exit_cfg  = REGIME_EXIT.get(regime, REGIME_EXIT[1])
    hold_days = STRUCTURAL_MIN_HOLD if severity == "Structural" else exit_cfg["min_hold"]
    e3_streak = STRUCTURAL_E3_STREAK if severity == "Structural" else E3_STREAK

    # Skip the minimum hold period — don't scan for exits too early after entry
    if len(series) > hold_days:
        series = series.iloc[hold_days:]

    # Step 1: find elevated confirmation (>= 50 for ELEVATED_CONFIRM days)
    elevated_confirmed = _first_streak(series >= C4_THRESH, ELEVATED_CONFIRM)
    if elevated_confirmed is None:
        return {
            "exit_confirmed": False,
            "exit_condition": None,
            "exit_candidate_date": None,
            "exit_confirmed_date": None,
            "false_exit_count": 0,
            "duration_model": None,
            "duration_actual": int((T_end - pd.Timestamp(entry_date)).days),
            "duration_delta": None,
            "exit_grade": "⚪ No Elevated State",
        }

    monitoring_series = series.loc[elevated_confirmed:]
    false_exit_count = 0
    search_from = 0

    while search_from < len(monitoring_series):
        remaining = monitoring_series.iloc[search_from:]
        if len(remaining) < E1_STREAK:
            break

        # Check exit candidates on remaining series
        candidate_date, condition = _find_exit_candidate(
            remaining,
            e1_thresh=exit_cfg["e1_thresh"],
            e1_streak=exit_cfg["e1_streak"],
            e3_streak=e3_streak,
        )
        if candidate_date is None:
            break

        # Confirmation window
        cand_pos = monitoring_series.index.get_loc(candidate_date)
        confirm_end_pos = min(len(monitoring_series), cand_pos + CONFIRM_WIN)
        confirm_window = monitoring_series.iloc[cand_pos: confirm_end_pos]

        if len(confirm_window) < CONFIRM_WIN:
            # Not enough data to confirm (end of dataset) — treat as tentative
            exit_confirmed_date = confirm_window.index[-1]
            duration_model = int((exit_confirmed_date - entry_date).days)
            duration_actual = int((T_end - entry_date).days)
            return {
                "exit_confirmed": True,
                "exit_condition": condition,
                "exit_candidate_date": candidate_date,
                "exit_confirmed_date": exit_confirmed_date,
                "false_exit_count": false_exit_count,
                "duration_model": duration_model,
                "duration_actual": duration_actual,
                "duration_delta": duration_model - duration_actual,
                "exit_grade": _exit_grade(exit_confirmed_date, T_end, false_exit_count),
            }

        # Did CSI stay below 50 for the full window?
        if (confirm_window < C4_THRESH).all():
            exit_confirmed_date = confirm_window.index[-1]
            duration_model = int((exit_confirmed_date - entry_date).days)
            duration_actual = int((T_end - entry_date).days)
            return {
                "exit_confirmed": True,
                "exit_condition": condition,
                "exit_candidate_date": candidate_date,
                "exit_confirmed_date": exit_confirmed_date,
                "false_exit_count": false_exit_count,
                "duration_model": duration_model,
                "duration_actual": duration_actual,
                "duration_delta": duration_model - duration_actual,
                "exit_grade": _exit_grade(exit_confirmed_date, T_end, false_exit_count),
            }
        else:
            # False exit — CSI bounced back above 50
            false_exit_count += 1
            # Resume search from where confirmation failed
            bounce_pos = (confirm_window >= C4_THRESH).values.argmax()
            search_from = cand_pos + bounce_pos + 1

    return {
        "exit_confirmed": False,
        "exit_condition": None,
        "exit_candidate_date": None,
        "exit_confirmed_date": None,
        "false_exit_count": false_exit_count,
        "duration_model": None,
        "duration_actual": int((T_end - entry_date).days),
        "duration_delta": None,
        "exit_grade": "🔴 No Confirmed Exit",
    }


def _find_exit_candidate(
    s: pd.Series,
    e1_thresh: float = E1_MEAN_THRESH,
    e1_streak: int = E1_STREAK,
    e3_streak: int = E3_STREAK,
) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    """Scan series for first E1/E2/E3 candidate. Returns (date, condition_name)."""
    candidates = {}

    # E1: rolling 21d mean < e1_thresh for e1_streak consecutive days
    mean_21 = s.rolling(21, min_periods=10).mean()
    e1 = _first_streak(mean_21 < e1_thresh, e1_streak)
    if e1 is not None:
        candidates["E1"] = _streak_start(mean_21 < e1_thresh, e1, e1_streak)

    # E2: drops >= E2_DROP pts in any 10-day window AND lands < E2_LANDING
    e2 = _any_drop(s, E2_DROP, E2_WIN, E2_LANDING)
    if e2 is not None:
        candidates["E2"] = e2

    # E3: crosses below 50 AND 10d rolling mean also below 50, held for e3_streak days
    mean_10 = s.rolling(10, min_periods=5).mean()
    e3_mask = (s < E3_THRESH) & (mean_10 < E3_THRESH)
    e3_confirmed = _first_streak(e3_mask, e3_streak)
    if e3_confirmed is not None:
        candidates["E3"] = _streak_start(e3_mask, e3_confirmed, e3_streak)

    if not candidates:
        return None, None
    earliest = min(candidates, key=lambda k: candidates[k])
    return candidates[earliest], earliest


def _exit_grade(exit_date: pd.Timestamp, T_end: pd.Timestamp, false_exits: int) -> str:
    delta_days = int((exit_date - T_end).days)
    if false_exits == 0 and abs(delta_days) <= 20:
        return "🟢 Clean"
    if false_exits == 0 and delta_days > 20:
        return "🟡 Extended"
    if false_exits == 0 and delta_days < -20:
        return "🟠 Premature"
    return f"🔴 False Exit ×{false_exits}"


# ─────────────────────────────────────────────────────────────────────────────
# Full lifecycle per episode
# ─────────────────────────────────────────────────────────────────────────────

def analyze_episode(
    pred_full: pd.Series,
    ep_name: str,
    T_start: str,
    T_end: str,
    regimes: Optional[pd.Series] = None,
) -> dict:
    """Run full entry + exit lifecycle analysis for a single episode."""
    # Resolve HMM regime at T_start
    regime = 1
    if regimes is not None:
        T = pd.Timestamp(T_start)
        idx = regimes.index.searchsorted(T)
        idx = min(idx, len(regimes) - 1)
        val = regimes.iloc[idx]
        regime = int(val) if not np.isnan(float(val)) else 1

    entry = detect_entry(pred_full, T_start, regime=regime)

    # Compute episode peak to determine severity tier
    ep_series = _slice(pred_full, T_start, T_end)
    peak_pred = float(ep_series.max()) if len(ep_series) > 0 else np.nan
    peak_date = ep_series.idxmax() if len(ep_series) > 0 else None
    severity  = "Structural" if (not np.isnan(peak_pred) and peak_pred >= ACUTE_PEAK_THRESH) else "Acute"

    entry_date = entry.get("signal_date") or pd.Timestamp(T_start)
    exit_info  = detect_exit(pred_full, entry_date, T_end, regime=regime, severity=severity)

    return {
        "episode":       ep_name,
        "T_start":       T_start,
        "T_end":         T_end,
        # Context
        "regime":        regime,
        "severity":      severity,
        # Entry
        "entry_hit":         entry["hit"],
        "entry_conditions":  ", ".join(entry["conditions_fired"]),
        "entry_signal_date": entry.get("signal_date"),
        "entry_lead_days":   entry.get("lead_days"),
        "entry_grade":       entry["grade"],
        "peak_in_win_a":     entry.get("peak_in_win_a"),
        "days_above_40_a":   entry.get("days_above_40_a"),
        # Episode peak
        "peak_pred_csi":     round(peak_pred, 1),
        "peak_date":         peak_date,
        # Exit
        "exit_confirmed":        exit_info["exit_confirmed"],
        "exit_condition":        exit_info["exit_condition"],
        "exit_candidate_date":   exit_info["exit_candidate_date"],
        "exit_confirmed_date":   exit_info["exit_confirmed_date"],
        "false_exit_count":      exit_info["false_exit_count"],
        "duration_model_days":   exit_info["duration_model"],
        "duration_actual_days":  exit_info["duration_actual"],
        "duration_delta_days":   exit_info["duration_delta"],
        "exit_grade":            exit_info["exit_grade"],
    }


def analyze_all_episodes(
    pred_df: pd.DataFrame,
    episodes: Dict[str, Tuple[str, str]],
    horizon: int,
    regimes: Optional[pd.Series] = None,
) -> List[dict]:
    """Run lifecycle analysis for all episodes. pred_df from predict_historical."""
    pred_full = pred_df["stress_score_pred"].dropna()
    results = []
    for ep_name, (T_start, T_end) in episodes.items():
        T = pd.Timestamp(T_start)
        if T < pred_full.index[0]:
            logger.debug("Skipping %s — before prediction history", ep_name)
            continue
        r = analyze_episode(pred_full, ep_name, T_start, T_end, regimes=regimes)
        r["horizon"] = horizon
        results.append(r)
        logger.info(
            "Episode %-25s | Regime %d | %s | Entry: %-22s | Exit: %s",
            ep_name, r["regime"], r["severity"], r["entry_grade"], r["exit_grade"],
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# False alarm rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_false_alarm_rate(
    pred_full: pd.Series,
    episodes: Dict[str, Tuple[str, str]],
    regimes: Optional[pd.Series] = None,
) -> dict:
    """
    Scan for C1/C2 signals that fire with no episode within PRE_WIN days.
    Uses regime-conditioned thresholds when regimes Series is provided.
    """
    protected = set()
    for _, (T_start, _) in episodes.items():
        T = pd.Timestamp(T_start)
        for dt in pred_full.index:
            if 0 <= (T - dt).days <= PRE_WIN:
                protected.add(dt)

    fa_dates = []
    step = 21
    for i in range(0, len(pred_full) - PRE_WIN, step):
        window = pred_full.iloc[i: i + PRE_WIN]
        if window.index[0] in protected:
            continue
        # Regime at this window's start date
        regime = 1
        if regimes is not None:
            idx = regimes.index.searchsorted(window.index[0])
            idx = min(idx, len(regimes) - 1)
            val = regimes.iloc[idx]
            regime = int(val) if not np.isnan(float(val)) else 1
        entry_cfg = REGIME_ENTRY.get(regime, REGIME_ENTRY[1])
        if (_check_c1(window, c1_thresh=entry_cfg["c1_thresh"]) is not None or
                _check_c2(pred_full, window, c2_pct=entry_cfg["c2_pct"]) is not None):
            fa_dates.append(window.index[0])

    years_covered = (pred_full.index[-1] - pred_full.index[0]).days / 365.25
    fa_per_year = round(len(fa_dates) / years_covered, 2) if years_covered > 0 else 0

    return {
        "n_false_alarms": len(fa_dates),
        "years_covered":  round(years_covered, 1),
        "fa_per_year":    fa_per_year,
        "fa_dates":       fa_dates[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_lifecycle_report(
    results: List[dict],
    horizon: int,
    fa_stats: Optional[dict] = None,
) -> None:
    """Print a formatted lifecycle report for all episodes at one horizon."""
    print(f"\n{'='*90}")
    print(f"  MARKET CRISIS SIGNAL — LIFECYCLE REPORT  |  Horizon: {horizon}d")
    print(f"{'='*90}")

    # Entry summary table
    print(f"\n  {'Episode':<26} {'Tier':<12} {'R':>2} {'Entry Grade':<26} {'Lead Days':>10} {'Conditions':<16} {'PeakA':>6}")
    print(f"  {'-'*98}")
    for r in results:
        lead     = str(r['entry_lead_days']) if r['entry_lead_days'] is not None else "—"
        peak     = f"{r['peak_in_win_a']:.0f}" if r['peak_in_win_a'] is not None and not (isinstance(r['peak_in_win_a'], float) and np.isnan(r['peak_in_win_a'])) else "—"
        tier     = r.get("severity", "—")
        regime_n = str(r.get("regime", "—"))
        print(f"  {r['episode']:<26} {tier:<12} {regime_n:>2} {r['entry_grade']:<26} {lead:>10} {r['entry_conditions']:<16} {peak:>6}")

    n_hit  = sum(1 for r in results if r["entry_hit"])
    n_3star = sum(1 for r in results if "⭐⭐⭐" in r["entry_grade"])
    n_2star = sum(1 for r in results if "⭐⭐" in r["entry_grade"] and "⭐⭐⭐" not in r["entry_grade"])
    n_1star = sum(1 for r in results if r["entry_grade"].startswith("⭐ "))
    n_rapid = sum(1 for r in results if "✓" in r["entry_grade"])
    n_miss  = sum(1 for r in results if "✗" in r["entry_grade"])
    avg_lead = np.mean([r["entry_lead_days"] for r in results
                        if r["entry_lead_days"] is not None and r["entry_lead_days"] > 0])

    n_structural = sum(1 for r in results if r.get("severity") == "Structural")
    n_acute      = sum(1 for r in results if r.get("severity") == "Acute")
    print(f"\n  Entry Hit Rate : {n_hit}/{len(results)} ({100*n_hit/len(results):.0f}%)")
    print(f"  Grade Breakdown: ⭐⭐⭐={n_3star}  ⭐⭐={n_2star}  ⭐={n_1star}  ✓={n_rapid}  ✗={n_miss}")
    print(f"  Severity       : Structural={n_structural}  Acute={n_acute}")
    print(f"  Avg Lead Time  : {avg_lead:.0f} days (pre-onset hits only)")

    # Exit summary table
    print(f"\n  {'Episode':<26} {'Exit Grade':<22} {'Cond':>5} {'False Exits':>12} {'Δ Duration':>12}")
    print(f"  {'-'*85}")
    for r in results:
        cond   = r["exit_condition"] or "—"
        fe     = str(r["false_exit_count"])
        delta  = f"{r['duration_delta_days']:+d}d" if r["duration_delta_days"] is not None else "—"
        print(f"  {r['episode']:<26} {r['exit_grade']:<22} {cond:>5} {fe:>12} {delta:>12}")

    n_clean     = sum(1 for r in results if "🟢" in r["exit_grade"])
    n_extended  = sum(1 for r in results if "🟡" in r["exit_grade"])
    n_premature = sum(1 for r in results if "🟠" in r["exit_grade"])
    n_false     = sum(1 for r in results if "🔴 False" in r["exit_grade"])
    n_no_exit   = sum(1 for r in results if "No Confirmed" in r["exit_grade"] or "No Elevated" in r["exit_grade"])
    avg_delta   = np.mean([r["duration_delta_days"] for r in results
                           if r["duration_delta_days"] is not None])

    print(f"\n  Exit Accuracy  : {n_clean + n_extended}/{len(results)} confirmed ({100*(n_clean+n_extended)/len(results):.0f}%)")
    print(f"  Grade Breakdown: 🟢={n_clean}  🟡={n_extended}  🟠={n_premature}  🔴={n_false}  ⚪/No={n_no_exit}")
    print(f"  Avg Duration Δ : {avg_delta:+.0f} days (+ = model held stress longer than labeled)")

    if fa_stats:
        print(f"\n  False Alarm Rate: {fa_stats['n_false_alarms']} alarms over {fa_stats['years_covered']} years "
              f"= {fa_stats['fa_per_year']}/year")

    print(f"\n{'='*90}")
