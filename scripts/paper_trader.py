"""Daily, close-to-close ML paper trader. No broker, credentials, or order execution."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.backtest import load_prices

FEATURES = ["ret_1", "ret_2", "ret_5", "sma5_gap", "sma10_gap"]
EXPOSURE = 0.25
COST_PER_UNIT_CHANGE = 0.0003
MAX_SESSION_LOSS = 0.02
MAX_DRAWDOWN = 0.10
STARTING_CAPITAL = 100_000.0

def feature_frame(close):
    return pd.DataFrame({
        "ret_1": close.pct_change(),
        "ret_2": close.pct_change(2),
        "ret_5": close.pct_change(5),
        "sma5_gap": close / close.rolling(5).mean() - 1,
        "sma10_gap": close / close.rolling(10).mean() - 1,
    }).replace([np.inf, -np.inf], np.nan)

def initial_state(last_date, last_close, capital=STARTING_CAPITAL):
    if capital <= 0 or last_close <= 0:
        raise ValueError("capital and last_close must be positive.")
    return {
        "version": 1, "mode": "daily_paper_trading", "risk_profile": "moderate",
        "balance": float(capital), "peak_balance": float(capital),
        "position": 0, "last_date": last_date, "last_close": float(last_close),
        "paused": False, "sessions_processed": 0, "trades": 0,
        "live_trading": False
    }

def process_latest(csv_path, state=None, capital=STARTING_CAPITAL):
    df = load_prices(csv_path)
    latest = df.iloc[-1]
    latest_date = pd.Timestamp(latest["date"]).isoformat()
    latest_close = float(latest["close"])
    if state is None:
        new_state = initial_state(latest_date, latest_close, capital)
        return new_state, {
            "status": "initialized", "date": latest_date, "close": latest_close,
            "message": "Baseline created. No simulated trade opened on initialization.",
            "live_trading": False
        }
    if state.get("live_trading") is not False or state.get("mode") != "daily_paper_trading":
        raise ValueError("State is not a recognized paper-only state.")
    if latest_date <= state["last_date"]:
        return state, {
            "status": "no_new_candle", "date": state["last_date"],
            "message": "No new daily candle; state unchanged.", "live_trading": False
        }

    old_position = int(state["position"])
    asset_return = latest_close / float(state["last_close"]) - 1.0
    gross_return = old_position * EXPOSURE * asset_return
    balance_before = float(state["balance"])
    balance = balance_before * (1.0 + gross_return)
    peak = max(float(state["peak_balance"]), balance)
    drawdown = 1.0 - balance / peak
    paused = bool(state.get("paused", False) or gross_return <= -MAX_SESSION_LOSS or drawdown >= MAX_DRAWDOWN)

    # Only train on outcomes already known before the latest candle.
    history = df.iloc[:-1].copy()
    close_hist = history["close"].astype(float)
    features = feature_frame(close_hist)
    next_return = close_hist.shift(-1) / close_hist - 1.0
    train = features.assign(target=(next_return > 0).astype(int)).dropna()
    current_features = feature_frame(df["close"].astype(float)).iloc[[-1]][FEATURES]
    probability = None
    proposed_position = 0
    if not paused and len(train) >= 100 and train["target"].nunique() == 2:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)
        )
        model.fit(train[FEATURES], train["target"])
        probability = float(model.predict_proba(current_features)[0, 1])
        proposed_position = 1 if probability > 0.55 else (-1 if probability < 0.45 else 0)

    target_position = 0 if paused else proposed_position
    change = abs(target_position - old_position)
    transaction_cost = COST_PER_UNIT_CHANGE * EXPOSURE * change
    net_return = gross_return - transaction_cost
    balance = balance_before * (1.0 + net_return)
    peak = max(peak, balance)
    drawdown = 1.0 - balance / peak
    if drawdown >= MAX_DRAWDOWN and target_position != 0:
        extra_close_cost = COST_PER_UNIT_CHANGE * EXPOSURE * abs(target_position)
        balance *= (1.0 - extra_close_cost)
        transaction_cost += extra_close_cost
        net_return = balance / balance_before - 1.0
        target_position = 0
        paused = True
        peak = max(peak, balance)
        drawdown = 1.0 - balance / peak

    new_state = dict(state)
    new_state.update({
        "balance": float(balance), "peak_balance": float(peak),
        "position": int(target_position), "last_date": latest_date,
        "last_close": latest_close, "paused": bool(paused),
        "sessions_processed": int(state.get("sessions_processed", 0)) + 1,
        "trades": int(state.get("trades", 0)) + (1 if change else 0),
        "live_trading": False
    })
    report = {
        "status": "processed", "date": latest_date, "close": latest_close,
        "probability_up": probability, "old_position": old_position,
        "new_position": int(target_position), "exposure_cap": EXPOSURE,
        "gross_return": float(gross_return), "assumed_cost": float(transaction_cost),
        "net_return": float(net_return), "balance": float(balance),
        "drawdown": float(drawdown), "paused": bool(paused),
        "risk_profile": "moderate", "live_trading": False,
        "warning": "Paper simulation only. Daily data may be delayed or revised. No broker or live orders."
    }
    return new_state, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/EURUSD=X.csv")
    parser.add_argument("--state", default="state/paper_state.json")
    parser.add_argument("--report", default="reports/daily_paper_report.json")
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL)
    args = parser.parse_args()
    state_path, report_path = Path(args.state), Path(args.report)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    new_state, report = process_latest(args.csv, state, args.capital)
    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
