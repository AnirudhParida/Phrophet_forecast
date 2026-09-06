"""
tests/test_evaluation.py
=========================
Unit tests for pipeline/evaluation.py.

Tests are pure-Python and do NOT require NeuralProphet or PyTorch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import (
    mae,
    rmse,
    mape,
    compute_metrics,
    train_holdout_split,
    format_evaluation_table,
)


# ---------------------------------------------------------------------------
# Metric function tests
# ---------------------------------------------------------------------------

class TestMetricFunctions:

    def test_mae_perfect_prediction(self):
        a = np.array([10.0, 20.0, 30.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_mae_known_value(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7 → mean = 7/3
        assert mae(actual, predicted) == pytest.approx(7 / 3, rel=1e-6)

    def test_rmse_known_value(self):
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.array([3.0, 4.0, 0.0])
        # sqrt((9 + 16 + 0) / 3) = sqrt(25/3)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(25 / 3), rel=1e-6)

    def test_mape_perfect_prediction(self):
        a = np.array([5.0, 10.0, 20.0])
        assert mape(a, a) == pytest.approx(0.0)

    def test_mape_near_zero_guard(self):
        """MAPE should not raise division-by-zero for near-zero actuals."""
        actual = np.array([0.0, 1.0, 2.0])
        predicted = np.array([0.1, 1.0, 2.0])
        result = mape(actual, predicted)
        assert np.isfinite(result), "MAPE returned inf/nan for near-zero actuals"

    def test_mape_returns_percentage(self):
        """MAPE result should be in percent (e.g. 10.0 means 10 %)."""
        actual = np.array([100.0, 100.0])
        predicted = np.array([90.0, 110.0])
        # |100-90|/100 = 0.1, |100-110|/100 = 0.1 → mean = 0.1 → ×100 = 10 %
        assert mape(actual, predicted) == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Holdout split tests
# ---------------------------------------------------------------------------

class TestTrainHoldoutSplit:

    def _make_df(self, n_rows: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            "ds": pd.date_range("2025-01-01", periods=n_rows, freq="D"),
            "cpu_pct": np.random.uniform(1, 10, n_rows),
            "memory_pct": np.random.uniform(70, 90, n_rows),
            "disk_pct": np.random.uniform(65, 76, n_rows),
        })

    def test_split_sizes(self):
        df = self._make_df(100)
        train, holdout = train_holdout_split(df, holdout_days=30, n_lags=7, n_forecasts=7)
        assert len(train) == 70
        assert len(holdout) == 30

    def test_holdout_is_trailing(self):
        """Holdout must be the LAST N rows (temporal order preserved)."""
        df = self._make_df(100)
        train, holdout = train_holdout_split(df, holdout_days=20, n_lags=7, n_forecasts=7)
        assert train["ds"].max() < holdout["ds"].min(), "Holdout overlaps with training set!"

    def test_excessive_holdout_raises(self):
        df = self._make_df(20)
        with pytest.raises(ValueError, match="holdout_days"):
            train_holdout_split(df, holdout_days=25, n_lags=7, n_forecasts=7)

    def test_insufficient_train_raises(self):
        df = self._make_df(30)
        with pytest.raises(ValueError, match="n_lags"):
            # holdout=25 → only 5 train rows, but n_lags+n_forecasts=14
            train_holdout_split(df, holdout_days=25, n_lags=7, n_forecasts=7)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_returns_all_keys(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 19.0, 31.0])
        result = compute_metrics(actual, predicted, "cpu_pct")
        assert set(result.keys()) == {"MAE", "RMSE", "WAPE", "MAPE"}

    def test_values_are_positive(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 19.0, 31.0])
        result = compute_metrics(actual, predicted, "cpu_pct")
        for k, v in result.items():
            assert v >= 0.0, f"{k} returned a negative value: {v}"


# ---------------------------------------------------------------------------
# format_evaluation_table tests
# ---------------------------------------------------------------------------

class TestFormatEvaluationTable:

    def test_output_is_string(self):
        results = {
            "cpu_pct": {"MAE": 0.5, "RMSE": 0.7, "MAPE": 3.2},
            "memory_pct": {"MAE": 1.2, "RMSE": 1.8, "MAPE": 1.5},
        }
        table = format_evaluation_table(results, "HYDUPINTAPP16", n_lags=7)
        assert isinstance(table, str)
        assert "HYDUPINTAPP16" in table
        assert "n_lags=7" in table
        assert "cpu_pct" in table
        assert "memory_pct" in table
