import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.backtest import load_prices, make_dataset, run_backtest

class BacktestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        returns = rng.normal(0.0002, 0.003, 260)
        close = 83 * np.cumprod(1 + returns)
        self.df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
            "open": close * (1 - 0.001), "high": close * 1.002,
            "low": close * 0.998, "close": close, "volume": 1000
        })
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prices.csv"
        self.df.to_csv(self.path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_feature_creation(self):
        loaded = load_prices(str(self.path))
        data = make_dataset(loaded)
        self.assertEqual(len(loaded), 260)
        self.assertTrue({"ret_1", "ret_2", "ret_5", "sma5_gap", "sma10_gap", "target", "next_return"}.issubset(data.columns))

    def test_backtest_is_out_of_sample_and_finite(self):
        result = run_backtest(str(self.path))
        self.assertGreater(result["train_rows"], 30)
        self.assertGreater(result["test_rows"], 15)
        self.assertFalse(result["live_trading"])
        for key in ("accuracy", "net_return", "max_drawdown", "ending_balance"):
            self.assertTrue(np.isfinite(result[key]))

    def test_rejects_bad_capital(self):
        with self.assertRaises(ValueError):
            run_backtest(str(self.path), capital=0)

if __name__ == "__main__":
    unittest.main()
