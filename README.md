# Forex Autopilot — v1.0 research build

A static GitHub Pages dashboard plus a scheduled, daily EUR/USD paper-trading simulation.

## Safety scope
- Paper trading only. The simulator sets `live_trading` to false and rejects a state marked otherwise.
- No broker integration, credentials, API keys, order routing, deposits, or real-money execution.
- This is not financial advice or evidence of a profitable strategy.
- Daily market data from yfinance is an unofficial Yahoo Finance interface and may be delayed, revised, unavailable, or subject to provider terms.

## Daily paper-trading schedule
- GitHub Actions schedule: weekdays at 05:30 Asia/Kolkata.
- It downloads up to five years of EUR/USD daily candles, runs the full unit-test suite, then processes the newest available daily candle.
- The first run initializes a ₹1,00,000 virtual baseline with no position. Later runs update the simulated position, equity and report when a new candle exists.
- Duplicate runs for the same candle are idempotent.
- State is saved to `state/paper_state.json`; each report is uploaded as a GitHub Actions artifact for 30 days.
- GitHub Actions scheduled jobs may be delayed or skipped during high load. In public repositories, schedules may be disabled after 60 days without repository activity. The schedule only works when this workflow is committed to the default branch.

## Moderate simulator risk assumptions
- Maximum notional exposure: 25% of virtual equity; no leverage.
- Signal: logistic regression probability above 0.55 = long, below 0.45 = short, otherwise flat.
- Assumed transaction cost: 3 basis points per unit of position change, applied to the 25% exposure.
- Circuit breaker: pause after a processed session loses 2% or peak-to-trough drawdown reaches 10%.
- These are research assumptions, not validated risk controls or guarantees. The daily job processes the latest available candle; a delayed/missed run can span multiple market sessions.

## Data and model
- Pair: EUR/USD (`EURUSD=X`), daily candles.
- Model features: 1-, 2-, and 5-bar returns; gaps from 5- and 10-bar simple moving averages.
- Model is trained only on historical rows whose next-candle outcome is already known; the latest candle is prediction-only.
- This is a simple logistic-regression research baseline. It does not prove predictive edge.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/download_data.py --symbol EURUSD=X --period 5y --interval 1d
python scripts/paper_trader.py --csv data/EURUSD=X.csv
```

## GitHub setup
1. Commit `.github/workflows/daily-paper-trading.yml` to the repository's default branch.
2. In Settings → Actions → General → Workflow permissions, allow read and write permissions (the workflow also requests `contents: write`).
3. Open Actions → Daily Forex paper trading to inspect runs and download the daily JSON report.
4. Keep the repository active; GitHub can disable scheduled workflows after 60 days of inactivity on public repositories.

## Limitations
- GitHub Pages is static hosting; it cannot host a continuously running Python process.
- This workflow is a scheduled daily research job, not a 24/7 bot. GitHub may delay or drop scheduled jobs under load.
- A missing or stale market-data candle results in no new trade for that run.
- No real trading is possible through this project.

## Version
v1.0 research build — scheduled daily paper simulation, persisted virtual state, model signal, moderate exposure cap, circuit breakers and unit tests.
