"""
tests/test_forecaster.py
=========================
Unit tests for the production forecasting upgrades in ServerMetricsForecaster.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from config.settings import CAP_VALUE, FLOOR_VALUE, GROWTH, METRIC_N_LAGS, METRICS
from pipeline.forecaster import ServerMetricsForecaster


@pytest.fixture
def mock_df():
    """Generate a clean synthetic 100-day dataframe for testing."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "cpu_pct": np.linspace(20, 80, 100),
        "memory_pct": np.linspace(30, 70, 100),
        "disk_pct": np.linspace(10, 50, 100),
        "disk_read_bytes": np.linspace(100, 1000, 100),
        "disk_write_bytes": np.linspace(200, 2000, 100),
    })


def test_resolve_n_lags():
    forecaster = ServerMetricsForecaster()
    # Dynamic per-metric lookup when no override is given
    assert forecaster._resolve_n_lags("cpu_pct") == METRIC_N_LAGS["cpu_pct"]
    assert forecaster._resolve_n_lags("memory_pct") == METRIC_N_LAGS["memory_pct"]
    assert forecaster._resolve_n_lags("disk_pct") == METRIC_N_LAGS["disk_pct"]
    assert forecaster._resolve_n_lags("disk_read_bytes") == METRIC_N_LAGS["disk_read_bytes"]
    assert forecaster._resolve_n_lags("disk_write_bytes") == METRIC_N_LAGS["disk_write_bytes"]

    # Override takes precedence
    assert forecaster._resolve_n_lags("cpu_pct", override_n_lags=14) == 14


def test_prepare_np_dataframe_adds_cap_and_floor(mock_df):
    forecaster = ServerMetricsForecaster()
    np_df = forecaster._prepare_np_dataframe(mock_df, "cpu_pct")

    assert "ds" in np_df.columns
    assert "y" in np_df.columns
    assert "memory_pct" in np_df.columns
    assert "disk_pct" in np_df.columns
    assert "disk_read_bytes" in np_df.columns
    assert "disk_write_bytes" in np_df.columns

    if GROWTH == "logistic":
        assert "cap" in np_df.columns
        assert "floor" in np_df.columns
        assert (np_df["cap"] == CAP_VALUE).all()
        assert (np_df["floor"] == FLOOR_VALUE).all()


def test_build_model_hyperparameters():
    forecaster = ServerMetricsForecaster()
    model = forecaster._build_model(n_lags=7)
    
    assert model.n_lags == 7
    assert model.config_trend.growth == GROWTH


def test_make_future_df_stage1_integration(mock_df):
    forecaster = ServerMetricsForecaster()
    np_df = forecaster._prepare_np_dataframe(mock_df, "cpu_pct")

    # Mock stage 1 future predictions
    future_dates = pd.date_range("2026-04-11", periods=30, freq="D")
    s1_mem = pd.DataFrame({"ds": future_dates, "yhat1": [65.0] * 30})
    s1_disk = pd.DataFrame({"ds": future_dates, "yhat1": [45.0] * 30})
    s1_rbytes = pd.DataFrame({"ds": future_dates, "yhat1": [500.0] * 30})
    s1_wbytes = pd.DataFrame({"ds": future_dates, "yhat1": [1000.0] * 30})
    stage1 = {
        "memory_pct": s1_mem,
        "disk_pct": s1_disk,
        "disk_read_bytes": s1_rbytes,
        "disk_write_bytes": s1_wbytes,
    }

    # Mock model's make_future_dataframe method
    mock_model = MagicMock()
    fut_df = pd.DataFrame({
        "ds": future_dates,
        "y": [np.nan] * 30,
        "memory_pct": [np.nan] * 30,
        "disk_pct": [np.nan] * 30,
        "disk_read_bytes": [np.nan] * 30,
        "disk_write_bytes": [np.nan] * 30,
    })
    mock_model.make_future_dataframe.return_value = fut_df

    fut_res = forecaster._make_future_df(
        model=mock_model,
        train_np_df=np_df,
        full_host_df=mock_df,
        target_metric="cpu_pct",
        n_lags=7,
        stage1_forecasts=stage1,
    )

    if GROWTH == "logistic":
        assert "cap" in fut_res.columns
        assert "floor" in fut_res.columns

    # Verify stage 1 values populated
    assert (fut_res["memory_pct"] == 65.0).all()
    assert (fut_res["disk_pct"] == 45.0).all()
    assert (fut_res["disk_read_bytes"] == 500.0).all()
    assert (fut_res["disk_write_bytes"] == 1000.0).all()
