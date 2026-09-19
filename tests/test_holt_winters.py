"""
tests/test_holt_winters.py
==========================
Unit tests for Holt-Winters Exponential Smoothing forecasting pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.holt_winters import (
    ESHoltWintersForecaster,
    HoltWintersForecaster,
    _clip_metric,
)


@pytest.fixture
def synthetic_csv_df():
    """Synthetic daily server metrics DataFrame (100 days)."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    t = np.arange(100)
    # Seasonal component with period 7
    seasonal = 5.0 * np.sin(2 * np.pi * t / 7.0)

    return pd.DataFrame({
        "ds": dates,
        "cpu_pct": np.clip(10.0 + 0.1 * t + seasonal + np.random.normal(0, 0.5, 100), 0, 100),
        "memory_pct": np.clip(70.0 + 0.05 * t + seasonal + np.random.normal(0, 0.5, 100), 0, 100),
        "disk_pct": np.clip(60.0 + 0.02 * t + np.random.normal(0, 0.2, 100), 0, 100),
        "disk_read_bytes": np.maximum(1000.0 + 10.0 * t + np.random.normal(0, 50, 100), 0),
        "disk_write_bytes": np.maximum(2000.0 + 20.0 * t + np.random.normal(0, 100, 100), 0),
    })


@pytest.fixture
def synthetic_es_df():
    """Synthetic minute-level ES metrics DataFrame (300 minutes)."""
    dates = pd.date_range("2026-09-01 00:00:00", periods=300, freq="1min")
    t = np.arange(300)
    seasonal = 4.0 * np.sin(2 * np.pi * t / 60.0)

    return pd.DataFrame({
        "ds": dates,
        "y": np.clip(15.0 + seasonal + np.random.normal(0, 0.5, 300), 0, 100),
    })


# ---------------------------------------------------------------------------
# Unit tests for clipping & utilities
# ---------------------------------------------------------------------------

def test_clip_metric():
    arr = np.array([-5.0, 50.0, 120.0])
    clipped_pct = _clip_metric(arr, "cpu_pct")
    assert clipped_pct[0] == 0.0
    assert clipped_pct[1] == 50.0
    assert clipped_pct[2] == 100.0

    clipped_bytes = _clip_metric(arr, "disk_read_bytes")
    assert clipped_bytes[0] == 0.0
    assert clipped_bytes[1] == 50.0
    assert clipped_bytes[2] == 120.0


# ---------------------------------------------------------------------------
# Unit tests for HoltWintersForecaster (CSV pipeline)
# ---------------------------------------------------------------------------

def test_hw_forecaster_fit_and_predict(synthetic_csv_df):
    forecaster = HoltWintersForecaster(seasonal_periods=7, n_forecasts=14)
    forecaster.fit(synthetic_csv_df, metrics=["cpu_pct", "memory_pct"])

    assert "cpu_pct" in forecaster.fitted_models
    assert "memory_pct" in forecaster.fitted_models

    forecasts = forecaster.predict(synthetic_csv_df, periods=14, metrics=["cpu_pct"])
    assert "cpu_pct" in forecasts
    fc_df = forecasts["cpu_pct"]

    assert len(fc_df) == 14
    assert set(fc_df.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert fc_df["ds"].iloc[0] == pd.to_datetime("2026-04-11")  # day after 100th date
    assert (fc_df["yhat"] >= 0).all() and (fc_df["yhat"] <= 100).all()
    assert (fc_df["yhat_lower"] <= fc_df["yhat"]).all()
    assert (fc_df["yhat_upper"] >= fc_df["yhat"]).all()


def test_hw_forecaster_evaluate(synthetic_csv_df):
    forecaster = HoltWintersForecaster(seasonal_periods=7)
    results = forecaster.evaluate(synthetic_csv_df, holdout_days=20, metrics=["cpu_pct", "disk_pct"])

    assert "cpu_pct" in results
    assert "disk_pct" in results
    for m in ["cpu_pct", "disk_pct"]:
        res = results[m]
        assert "MAE" in res and res["MAE"] >= 0.0
        assert "RMSE" in res and res["RMSE"] >= 0.0
        assert "WAPE" in res and res["WAPE"] >= 0.0
        assert "MAPE" in res and res["MAPE"] >= 0.0


def test_hw_forecaster_save_and_load(synthetic_csv_df, tmp_path):
    forecaster = HoltWintersForecaster(seasonal_periods=7)
    forecaster.fit(synthetic_csv_df, metrics=["cpu_pct"])
    forecaster.save(dir_path=tmp_path, host_alias="TESTHOST")

    model_file = tmp_path / "TESTHOST_cpu_pct_hw.joblib"
    assert model_file.exists()

    new_forecaster = HoltWintersForecaster()
    new_forecaster.load(dir_path=tmp_path, host_alias="TESTHOST", metrics=["cpu_pct"])
    assert "cpu_pct" in new_forecaster.fitted_models

    fc = new_forecaster.predict(synthetic_csv_df, periods=7, metrics=["cpu_pct"])
    assert len(fc["cpu_pct"]) == 7


def test_hw_forecaster_fallback_short_series():
    # Only 10 points with seasonal_periods=7 (< 14 points needed)
    short_series = pd.Series(np.arange(10, dtype=float))
    forecaster = HoltWintersForecaster(seasonal_periods=7)
    fitted = forecaster.fit_single_metric(short_series, "short_metric")
    assert fitted is not None
    fc = fitted.forecast(3)
    assert len(fc) == 3


# ---------------------------------------------------------------------------
# Unit tests for ESHoltWintersForecaster (Elasticsearch pipeline)
# ---------------------------------------------------------------------------

def test_es_hw_forecaster_fit_and_predict(synthetic_es_df):
    forecaster = ESHoltWintersForecaster(resolution="1min", n_forecasts=30, seasonal_periods=60)
    forecaster.fit(synthetic_es_df)
    assert forecaster.fitted_model is not None

    fc_df = forecaster.predict(synthetic_es_df, periods=30)
    assert len(fc_df) == 30
    assert set(fc_df.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert (fc_df["yhat"] >= 0).all() and (fc_df["yhat"] <= 100).all()


def test_es_hw_forecaster_evaluate(synthetic_es_df):
    forecaster = ESHoltWintersForecaster(resolution="1min", seasonal_periods=60)
    res = forecaster.evaluate(synthetic_es_df, holdout_steps=60)

    assert "MAE" in res and res["MAE"] >= 0.0
    assert "RMSE" in res and res["RMSE"] >= 0.0
    assert "WAPE" in res and res["WAPE"] >= 0.0


def test_es_hw_forecaster_capacity(synthetic_es_df):
    forecaster = ESHoltWintersForecaster(resolution="1min", seasonal_periods=60)
    # Test 1 day capacity for fast test
    cap = forecaster.predict_capacity(synthetic_es_df, forecast_days=1)

    assert "minutely" in cap
    assert "hourly" in cap
    assert "daily" in cap
    assert len(cap["hourly"]) >= 24
    assert len(cap["daily"]) in [1, 2]
    assert "cpu_forecast_mean" in cap["hourly"].columns
    assert "cpu_forecast_min" in cap["hourly"].columns
    assert "cpu_forecast_max" in cap["hourly"].columns


def test_es_hw_forecaster_save_and_load(synthetic_es_df, tmp_path):
    forecaster = ESHoltWintersForecaster(resolution="1min", seasonal_periods=60)
    forecaster.fit(synthetic_es_df)

    save_path = tmp_path / "es_hw_test.joblib"
    forecaster.save(save_path)
    assert save_path.exists()

    new_fc = ESHoltWintersForecaster(resolution="1min")
    new_fc.load(save_path)
    pred = new_fc.predict(synthetic_es_df, periods=10)
    assert len(pred) == 10
