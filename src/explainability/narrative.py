"""
LLM narrative generator (Claude API) for stress report explanations.

When USE_LLM=False, generates structured rule-based text instead.
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _rule_based_narrative(
    predictions: Dict[int, dict],
    attribution_5d: List[dict],
    attribution_21d: List[dict],
    attribution_63d: List[dict],
    csi_history_stats: dict,
) -> str:
    """Fallback: generate structured text without LLM."""
    lines = []

    for h, attr in [(5, attribution_5d), (21, attribution_21d), (63, attribution_63d)]:
        pred = predictions.get(h)
        if not pred:
            continue
        score = pred.get("stress_score", "N/A")
        cls   = pred.get("stress_class", "N/A")
        delta = pred.get("stress_delta")
        direction = pred.get("market_direction", "")

        horizon_label = {5: "Short-Term (5-Day)", 21: "Medium-Term (21-Day)", 63: "Long-Term (63-Day)"}[h]
        lines.append(f"### {horizon_label} Outlook")
        lines.append(
            f"Predicted stress score: **{score}/100** ({cls}). "
            + (f"Change from current: **{delta:+.1f}**. " if delta is not None else "")
            + (f"Market direction signal: **{direction}**." if direction else "")
        )

        if attr:
            top = attr[:3]
            factor_strs = ", ".join(
                f"{f['label']} ({f['dimension']}"
                + (f", {f['percentile']:.0f}th pct" if f.get("percentile") else "")
                + ")"
                for f in top
            )
            lines.append(f"Primary drivers: {factor_strs}.")
        lines.append("")

    return "\n".join(lines)


def _llm_narrative(
    client,
    model: str,
    max_tokens: int,
    predictions: Dict[int, dict],
    attribution_5d: List[dict],
    attribution_21d: List[dict],
    attribution_63d: List[dict],
    csi_history_stats: dict,
) -> str:
    """Call Claude API to generate expert narrative."""

    def fmt_attr(attr):
        return "\n".join(
            f"  - {f['label']} ({f['dimension']}): SHAP={f['shap_value']:.3f}"
            + (f", current={f['current_value']}" if f.get("current_value") else "")
            + (f", {f['percentile']:.0f}th pct" if f.get("percentile") else "")
            for f in attr[:5]
        )

    def fmt_pred(h):
        p = predictions.get(h, {})
        return (
            f"  Stress Score: {p.get('stress_score', 'N/A')}/100 ({p.get('stress_class', 'N/A')}), "
            f"Delta: {p.get('stress_delta', 'N/A'):+.1f}, "
            f"Market Direction: {p.get('market_direction', 'N/A')} "
            f"(Down:{p.get('market_dir_proba', {}).get('Down', 'N/A')}, "
            f"Neutral:{p.get('market_dir_proba', {}).get('Neutral', 'N/A')}, "
            f"Up:{p.get('market_dir_proba', {}).get('Up', 'N/A')})"
        ) if p else "  Not available"

    prompt = f"""You are a senior cross-asset financial strategist writing the daily market stress briefing.

Current CSI (Composite Stress Index) Context:
- Current Stress Score: {predictions.get(5, {}).get('current_csi', 'N/A')}/100
- 1Y Average: {csi_history_stats.get('mean_1y', 'N/A'):.1f}, 1Y High: {csi_history_stats.get('max_1y', 'N/A'):.1f}

Predictions:
SHORT-TERM (5-Day):
{fmt_pred(5)}
Top SHAP Drivers:
{fmt_attr(attribution_5d)}

MEDIUM-TERM (21-Day):
{fmt_pred(21)}
Top SHAP Drivers:
{fmt_attr(attribution_21d)}

LONG-TERM (63-Day):
{fmt_pred(63)}
Top SHAP Drivers:
{fmt_attr(attribution_63d)}

Write a concise 3-paragraph stress briefing:
1. Current regime assessment and key risks (2-3 sentences)
2. Short/medium-term outlook with the most important factor explanations (3-4 sentences)
3. Long-term structural view and risk monitoring priorities (2-3 sentences)

Use clear, professional financial language. Reference specific indicators where relevant.
Do not repeat numbers verbatim from the data — synthesize and interpret."""

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_narrative(
    predictions: Dict[int, dict],
    attribution_5d: List[dict],
    attribution_21d: List[dict],
    attribution_63d: List[dict],
    csi: "pd.DataFrame",
    use_llm: bool = False,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-7",
    max_tokens: int = 1024,
) -> str:
    """
    Generate the narrative explanation for the stress report.

    Parameters
    ----------
    use_llm  : if True, calls Claude API; falls back to rule-based on error
    api_key  : Anthropic API key (or reads ANTHROPIC_API_KEY env var)
    """
    # Compute trailing CSI stats for context
    csi_comp = csi["csi_composite"].dropna()
    csi_history_stats = {
        "mean_1y": float(csi_comp.tail(252).mean()),
        "max_1y":  float(csi_comp.tail(252).max()),
        "min_1y":  float(csi_comp.tail(252).min()),
    }

    if not use_llm:
        return _rule_based_narrative(
            predictions, attribution_5d, attribution_21d, attribution_63d, csi_history_stats
        )

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        logger.warning("ANTHROPIC_API_KEY not set. Falling back to rule-based narrative.")
        return _rule_based_narrative(
            predictions, attribution_5d, attribution_21d, attribution_63d, csi_history_stats
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        return _llm_narrative(
            client, model, max_tokens,
            predictions, attribution_5d, attribution_21d, attribution_63d,
            csi_history_stats,
        )
    except Exception as exc:
        logger.warning("Claude API call failed (%s). Falling back to rule-based narrative.", exc)
        return _rule_based_narrative(
            predictions, attribution_5d, attribution_21d, attribution_63d, csi_history_stats
        )
