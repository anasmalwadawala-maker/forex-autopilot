"""Download a public Yahoo Finance history sample via yfinance (unofficial interface)."""
from pathlib import Path
import argparse
import yfinance as yf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="USDINR=X")
    p.add_argument("--period", default="5y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--output-dir", default="data")
    a = p.parse_args()
    if a.interval not in {"1d", "1wk", "1mo"}:
        raise SystemExit("Only daily/weekly/monthly intervals are supported in this workflow.")
    frame = yf.download(a.symbol, period=a.period, interval=a.interval, auto_adjust=False, progress=False)
    if frame.empty:
        raise SystemExit("No data returned. Try another period or check provider availability.")
    if hasattr(frame.columns, "levels"):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    date_col = "date" if "date" in frame else "datetime"
    frame = frame.rename(columns={date_col: "date"})
    needed = ["date", "open", "high", "low", "close"]
    frame = frame[needed + (["volume"] if "volume" in frame else [])].dropna(subset=needed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.symbol.replace('=', '_').replace('^', '')}.csv"
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame)} rows to {path}")
    print("Provider note: yfinance is an unofficial Yahoo Finance interface; availability, coverage and terms may change.")

if __name__ == "__main__":
    main()
