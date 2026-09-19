"""
tests/test_chronos.py
=====================
Unit and integration tests for Amazon Chronos-2 universal foundation model pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cli.cli_chronos import build_chronos_parser
from pipeline.models.chronos.chronos_csv import ChronosCSVForecaster, _clip_metric
from pipeline.models.chronos.chronos_es import ESChronosForecaster
from pipeline.models.chronos.model_loader import resolve_device


def test_resolve_device():
    """Verify hardware resolution logic for auto, cpu, and cuda."""
    assert resolve_device("cpu") == "cpu"
    auto_dev = resolve_device("auto")
    assert auto_dev in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert resolve_device("cuda") == "cpu"
    else:
        assert resolve_device("cuda") == "cuda"


def test_clip_metric():
    """Verify metric clipping to realistic physical bounds."""
    # Percentage clipping
    vals = np.array([-10.0, 50.0, 115.0])
    clipped_pct = _clip_metric(vals, "cpu_pct")
    np.testing.assert_array_equal(clipped_pct, [0.0, 50.0, 100.0])

    # Bytes clipping
    bytes_vals = np.array([-1024.0, 0.0, 4096.0])
    clipped_bytes = _clip_metric(bytes_vals, "disk_write_bytes")
    np.testing.assert_array_equal(clipped_bytes, [0.0, 0.0, 4096.0])


def test_chronos_csv_multivariate_predict():
    """Verify multivariate zero-shot forecast with group attention across multiple host metrics."""
    dates = pd.date_range("2026-01-01", periods=48, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates,
            "cpu_pct": np.sin(np.linspace(0, 8, 48)) * 20.0 + 50.0,
            "memory_pct": np.cos(np.linspace(0, 8, 48)) * 15.0 + 60.0,
            "disk_pct": np.linspace(40.0, 60.0, 48),
        }
    )

    forecaster = ChronosCSVForecaster(device="cpu", n_forecasts=7, multivariate=True)
    forecaster.fit(df, host_alias="TEST_HOST")
    fc = forecaster.predict(df, periods=7, metrics=["cpu_pct", "memory_pct", "disk_pct"])

    assert "cpu_pct" in fc
    assert "memory_pct" in fc
    assert "disk_pct" in fc

    for metric in ["cpu_pct", "memory_pct", "disk_pct"]:
        res_df = fc[metric]
        assert len(res_df) == 7
        assert list(res_df.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
        assert (res_df["yhat"] >= 0.0).all() and (res_df["yhat"] <= 100.0).all()
        assert (res_df["yhat_lower"] <= res_df["yhat_upper"]).all()


def test_chronos_csv_univariate_predict():
    """Verify univariate forecast mode (isolated series projection)."""
    dates = pd.date_range("2026-01-01", periods=48, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates,
            "cpu_pct": np.sin(np.linspace(0, 8, 48)) * 20.0 + 50.0,
        }
    )

    forecaster = ChronosCSVForecaster(device="cpu", n_forecasts=5, multivariate=False)
    forecaster.fit(df, host_alias="TEST_HOST")
    fc = forecaster.predict(df, periods=5, metrics=["cpu_pct"])

    assert "cpu_pct" in fc
    assert len(fc["cpu_pct"]) == 5
    assert (fc["cpu_pct"]["yhat"] >= 0.0).all() and (fc["cpu_pct"]["yhat"] <= 100.0).all()


def test_chronos_csv_evaluate():
    """Verify holdout evaluation on CSV data returns expected metrics."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates,
            "cpu_pct": np.linspace(25.0, 45.0, 50),
        }
    )

    forecaster = ChronosCSVForecaster(device="cpu", multivariate=False)
    results = forecaster.evaluate(df, test_days=7, metrics=["cpu_pct"])

    assert "cpu_pct" in results
    scores = results["cpu_pct"]
    assert "mae" in scores
    assert "rmse" in scores
    assert "wape" in scores
    assert "mape" in scores
    assert scores["mae"] >= 0.0


def test_chronos_csv_plot_and_save(tmp_path):
    """Verify plot and forecast saving functions."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "cpu_pct": np.sin(np.linspace(0, 6, 40)) * 15.0 + 45.0,
    })

    forecaster = ChronosCSVForecaster(device="cpu", n_forecasts=6)
    fc = forecaster.predict(df, periods=6, metrics=["cpu_pct"])

    chart = forecaster.plot_forecast(df, fc, host_alias="TEST_PLOT", output_dir=tmp_path)
    assert chart.exists()

    csv_path, xlsx_path = forecaster.save_forecast(fc, host_alias="TEST_PLOT", output_dir=tmp_path)
    assert csv_path.exists()
    assert xlsx_path.exists()


def test_chronos_es_predict_and_evaluate():
    """Verify ES forecaster for hourly streaming metrics."""
    dates = pd.date_range("2026-01-01", periods=48, freq="1h")
    df = pd.DataFrame(
        {
            "ds": dates,
            "y": np.clip(np.sin(np.linspace(0, 8, 48)) * 12.0 + 35.0, 0.0, 100.0),
        }
    )

    forecaster = ESChronosForecaster(resolution="1h", device="cpu", n_forecasts=6)
    forecaster.fit(df)
    fc = forecaster.predict(df, periods=6)

    assert len(fc) == 6
    assert list(fc.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert (fc["yhat"] >= 0.0).all() and (fc["yhat"] <= 100.0).all()

    eval_res = forecaster.evaluate(df, test_steps=6)
    assert eval_res["MAE"] >= 0.0
    assert eval_res["RMSE"] >= 0.0


def test_chronos_es_capacity_and_plots(tmp_path):
    """Verify 30-day capacity forecast, daily aggregation, and chart generation."""
    dates = pd.date_range("2026-01-01", periods=96, freq="1h")
    df = pd.DataFrame(
        {
            "ds": dates,
            "y": np.clip(np.sin(np.linspace(0, 12, 96)) * 15.0 + 40.0, 0.0, 100.0),
        }
    )

    forecaster = ESChronosForecaster(resolution="1h", device="cpu")
    cap = forecaster.predict_capacity(df, forecast_days=5)

    assert "hourly" in cap
    assert "daily" in cap
    assert len(cap["hourly"]) == 5 * 24
    assert len(cap["daily"]) >= 5

    charts = forecaster.plot_capacity_forecast(
        df,
        cap,
        entity_id="test_entity",
        output_dir=tmp_path,
        forecast_days=5,
    )

    assert "hourly" in charts and charts["hourly"].exists()
    assert "daily" in charts and charts["daily"].exists()

    paths = forecaster.save_capacity_forecast(cap, entity_id="test_entity", output_dir=tmp_path)
    assert "hourly" in paths and paths["hourly"][0].exists() and paths["hourly"][1].exists()
    assert "daily" in paths and paths["daily"][0].exists() and paths["daily"][1].exists()


def test_chronos_cli_parser():
    """Verify CLI parser options for CSV and ES subcommands."""
    parser = build_chronos_parser()

    # CSV predict arguments
    args_csv = parser.parse_args([
        "csv", "predict", "--host", "HYDUPINTAPP16", "--periods", "14", "--device", "cpu"
    ])
    assert args_csv.pipeline == "csv"
    assert args_csv.command == "predict"
    assert args_csv.host == "HYDUPINTAPP16"
    assert args_csv.periods == 14
    assert args_csv.device == "cpu"
    assert args_csv.univariate is False

    # CSV univariate predict arguments
    args_uni = parser.parse_args([
        "csv", "predict", "--host", "HYDUPINTAPP16", "--univariate"
    ])
    assert args_uni.univariate is True

    # ES capacity arguments
    args_es = parser.parse_args([
        "es", "capacity", "--forecast-days", "30", "--device", "cpu"
    ])
    assert args_es.pipeline == "es"
    assert args_es.command == "capacity"
    assert args_es.forecast_days == 30
    assert args_es.device == "cpu"
