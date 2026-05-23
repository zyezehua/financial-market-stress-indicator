# Global Market Stress Indicator

A machine learning system for predicting cross-asset financial market stress levels across multiple time horizons, with factor attribution and automated report generation.

## Architecture

```
Data (Yahoo Finance + FRED) → Feature Engineering (6 dimensions) → CSI Label Construction
→ Ensemble Model (LightGBM + XGBoost + Ridge) → SHAP Attribution → Claude API Narrative
→ PDF Report + Interactive HTML Dashboard
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
```

Edit `.env`:
- `FRED_API_KEY` — Free at https://fred.stlouisfed.org/docs/api/api_key.html (**required**)
- `ANTHROPIC_API_KEY` — Optional, only for LLM-generated narrative

### 3. Train models (first run, ~10-20 min)
```bash
python scripts/train_model.py --validate
```

### 4. Run daily report
```bash
python scripts/daily_run.py
```

Reports are saved to `reports/`.

## Usage

```bash
# Full daily run (HTML + PDF)
python scripts/daily_run.py

# Enable Claude API narrative
python scripts/daily_run.py --use-llm

# Run for a specific historical date
python scripts/daily_run.py --date 2024-10-15

# Backtest on COVID episode
python scripts/backtest.py --episode covid

# Retrain single horizon
python scripts/train_model.py --horizon 21
```

## Composite Stress Index (CSI)

The CSI aggregates stress across 6 dimensions, each normalized to 0–100:

| Dimension | Weight | Key Signals |
|---|---|---|
| Equity | 20% | VIX, VIX term structure, SPY realized vol, drawdown |
| Credit | 20% | HY/IG OAS spreads, CDX proxy, HYG vol |
| Rates | 17.5% | Yield curve inversion, TLT realized vol, rate velocity |
| Liquidity | 17.5% | SOFR spread, cross-asset correlation, contagion |
| FX | 12.5% | USD strength, JPY safe-haven demand, FX vol |
| International | 12.5% | EM vol, Europe/Japan, Gold safe-haven |

**Classification:**
- 0–25: Low
- 26–50: Elevated
- 51–75: High
- 76–100: Extreme

## Prediction Outputs

For each horizon (5d / 21d / 63d):
- Stress score (0–100) + classification
- Stress delta (direction & magnitude of expected change)
- Market direction signal: Up / Neutral / Down (5d, 21d only)
- Direction probability distribution

## Phase Roadmap

- **Phase 1** (current): US core assets, 3 horizons, SHAP + rule-based narrative
- **Phase 2**: Short/medium direction prediction refinement, Claude API narrative
- **Phase 3**: International sub-models, enhanced dashboard
