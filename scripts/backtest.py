"""Chronological logistic-regression paper backtest. No broker integration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED = {"date", "open", "high", "low", "close"}

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if (df["close"] <= 0).any() or len(df) < 100:
        raise ValueError("Need at least 100 valid rows with positive closing prices.")
    return df

def make_dataset(df: pd.DataFrame):
    close = df["close"].astype(float)
    feat = pd.DataFrame({
        "ret_1": close.pct_change(),
        "ret_2": close.pct_change(2),
        "ret_5": close.pct_change(5),
        "sma5_gap": close / close.rolling(5).mean() - 1,
        "sma10_gap": close / close.rolling(10).mean() - 1,
    })
    # Features at t predict the direction of the next candle (t -> t+1).
    next_return = close.shift(-1) / close - 1
    data = feat.assign(target=(next_return > 0).astype(int), next_return=next_return)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data

def run_backtest(csv_path: str, capital: float = 100000.0, cost_pct: float = 0.05):
    if capital <= 0 or not 0 <= cost_pct <= 2:
        raise ValueError("capital must be positive and cost_pct between 0 and 2.")
    df = load_prices(csv_path)
    data = make_dataset(df)
    split = int(len(data) * 0.70)
    if split < 30 or len(data) - split < 15:
        raise ValueError("Insufficient rows for chronological train/test split.")
    features = ["ret_1", "ret_2", "ret_5", "sma5_gap", "sma10_gap"]
    train, test = data.iloc[:split], data.iloc[split:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7))
    model.fit(train[features], train["target"])
    prob = model.predict_proba(test[features])[:, 1]
    prediction = (prob >= 0.5).astype(int)
    # Long/short/flat paper positions; a position earns the next-candle return.
    position = np.where(prob > 0.55, 1, np.where(prob < 0.45, -1, 0))
    returns = test["next_return"].to_numpy()
    changes = np.abs(np.diff(np.r_[0, position]))
    costs = (cost_pct / 100.0) * changes
    strategy_returns = position * returns - costs
    equity = np.cumprod(1 + strategy_returns)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = 1 - equity / peaks
    accuracy = float(accuracy_score(test["target"], prediction))
    try:
        auc = float(roc_auc_score(test["target"], prob))
    except ValueError:
        auc = None
    result = {
        "symbol": "USDINR=X", "mode": "historical_paper_backtest",
        "rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "accuracy": accuracy, "roc_auc": auc,
        "net_return": float(equity[-1] - 1),
        "max_drawdown": float(drawdown.max(initial=0)),
        "starting_balance": float(capital), "ending_balance": float(capital * equity[-1]),
        "position_changes": int(np.count_nonzero(changes)),
        "assumed_cost_pct_per_position_change": float(cost_pct),
        "first_test_date": str(df.loc[data.index[split], "date"].date()),
        "last_test_date": str(df.loc[data.index[-1], "date"].date()),
        "live_trading": False,
        "warning": "Historical simulation only. Not a forecast, investment recommendation, or executable trading signal."
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/USDINR=X.csv")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--cost-pct", type=float, default=0.05)
    parser.add_argument("--output", default="reports/backtest.json")
    args = parser.parse_args()
    result = run_backtest(args.csv, args.capital, args.cost_pct)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
