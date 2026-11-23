# Forex Autopilot — v0.3

A static GitHub Pages research dashboard plus a Python historical-data and machine-learning backtest pipeline.

## Safety scope
- Historical analysis and paper simulations only.
- No broker integration, credentials, API keys, order routing or real-money execution.
- Backtest outputs are not forecasts or investment advice.
- Indian users must independently verify current RBI/FEMA rules and use only permitted, authorised routes before considering any financial activity. This repository does not enable live trading.

## GitHub Pages
1. Open **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select branch **main**, folder **/(root)**, then Save.
4. Wait for the Pages URL to appear in Settings → Pages.

## Dashboard CSV format
Upload a CSV locally in the dashboard. Required columns: `date,open,high,low,close`; `volume` optional. Use at least 80 rows for the browser experiment and sort oldest to newest. Data stays in your browser; it is not uploaded.

## Python research workflow
1. Open **Actions → Historical research backtest → Run workflow**.
2. Default symbol is `USDINR=X`; choose daily/weekly/monthly interval.
3. The workflow installs requirements, runs unit tests, downloads public history through `yfinance`, trains a logistic-regression classifier on the first 70% of observations, and evaluates the final 30% chronologically.
4. Download the report artifact from the completed workflow run.

The Yahoo Finance interface used by yfinance is unofficial. Data availability, coverage, licensing and provider terms may change. Verify data suitability and rights before use. Historical USD/INR data is not a live executable quote.

## Model notes
Features: one-, two- and five-bar returns, and gaps from five- and ten-bar moving averages. Each feature row at time *t* predicts the direction of the next bar. The Python backtest uses a chronological holdout, standardized logistic regression, long/short/flat paper positions, and a configurable simplified cost assumption. This is a research baseline, not evidence of a profitable strategy. Expect weak or negative results.

## Local run
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/download_data.py --symbol USDINR=X --period 5y --interval 1d
python scripts/backtest.py --csv data/USDINR_X.csv
```

The browser dashboard uses a small JavaScript logistic-regression implementation and runs only on the CSV selected by the visitor. GitHub Pages is static hosting; it cannot keep a Python process running continuously. GitHub Actions runs only when manually triggered here.

## Version
v0.3 — local CSV backtest, Python ML research workflow, unit tests and static dashboard.
