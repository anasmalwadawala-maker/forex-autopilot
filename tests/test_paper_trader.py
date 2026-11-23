import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.paper_trader import process_latest

class PaperTraderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        close = 1.08 * np.cumprod(1 + rng.normal(0.0001, 0.002, 260))
        self.df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=260, freq="B"),
            "open": close, "high": close * 1.002,
            "low": close * 0.998, "close": close
        })
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prices.csv"
        self.df.to_csv(self.path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_initialization_is_flat_and_paper_only(self):
        state, report = process_latest(str(self.path))
        self.assertEqual(state["position"], 0)
        self.assertFalse(state["live_trading"])
        self.assertEqual(report["status"], "initialized")

    def test_duplicate_candle_is_idempotent(self):
        state, _ = process_latest(str(self.path))
        again, report = process_latest(str(self.path), state)
        self.assertEqual(again, state)
        self.assertEqual(report["status"], "no_new_candle")

    def test_new_candle_updates_state_without_live_trading(self):
        state, _ = process_latest(str(self.path))
        extra = self.df.iloc[[-1]].copy()
        extra["date"] = extra["date"] + pd.Timedelta(days=1)
        extra["close"] *= 1.001
        extra["open"] = extra["close"]
        extra["high"] = extra["close"] * 1.002
        extra["low"] = extra["close"] * 0.998
        extended = pd.concat([self.df, extra], ignore_index=True)
        extended.to_csv(self.path, index=False)
        updated, report = process_latest(str(self.path), state)
        self.assertEqual(report["status"], "processed")
        self.assertFalse(updated["live_trading"])
        self.assertLessEqual(abs(updated["position"]), 1)
        self.assertTrue(np.isfinite(updated["balance"]))

    def test_rejects_non_paper_state(self):
        state, _ = process_latest(str(self.path))
        state["live_trading"] = True
        with self.assertRaises(ValueError):
            process_latest(str(self.path), state)

if __name__ == "__main__":
    unittest.main()
