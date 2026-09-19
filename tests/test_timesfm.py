"""
tests/test_timesfm.py
=====================
Unit and integration tests for Google TimesFM 3.0 foundation model pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cli.cli_timesfm import build_tfm_parser
from pipeline.models.timesfm.model_loader import resolve_device
from pipeline.models.timesfm.tfm_csv import TimesFMForecaster, _clip_metric
from pipeline.models.timesfm.tfm_es import ESTimesFMForecaster


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
    vals = np.array([-15.0, 45.0, 120.0])
    clipped_pct = _clip_metric(vals, "cpu_pct")
    np.testing.assert_array_equal(clipped_pct, [0.0, 45.0, 100.0])

    # Bytes clipping
    bytes_vals = np.array([-500.0, 0.0, 2048.0])
    clipped_bytes = _clip_metric(bytes_vals, "disk_read_bytes")
    np.testing.assert_array_equal(clipped_bytes, [0.0, 0.0, 2048.0])


def test_timesfm_csv_forecaster_predict():
    """Verify zero-shot forecast generation on multi-metric CSV host data."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates,
            "cpu_pct": np.sin(np.linspace(0, 10, 60)) * 20.0 + 50.0,
            "memory_pct": np.cos(np.linspace(0, 10, 60)) * 15.0 + 60.0,
        }
    )

    forecaster = TimesFMForecaster(device="cpu", n_forecasts=7)
    forecaster.fit(df, host_alias="TESTHOST")
    fc = forecaster.predict(df, periods=7, metrics=["cpu_pct", "memory_pct"])

    assert "cpu_pct" in fc
    assert "memory_pct" in fc

    for metric in ["cpu_pct", "memory_pct"]:
        res_df = fc[metric]
        assert len(res_df) == 7
        assert list(res_df.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
        assert (res_df["yhat"] >= 0.0).all() and (res_df["yhat"] <= 100.0).all()
        assert (res_df["yhat_lower"] <= res_df["yhat_upper"]).all()


def test_timesfm_csv_forecaster_evaluate():
    """Verify holdout evaluation on CSV data."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "ds": dates,
            "cpu_pct": np.linspace(20.0, 40.0, 50),
        }
    )

    forecaster = TimesFMForecaster(device="cpu")
    results = forecaster.evaluate(df, holdout_days=7, metrics=["cpu_pct"])

    assert "cpu_pct" in results
    scores = results["cpu_pct"]
    assert "MAE" in scores
    assert "RMSE" in scores
    assert "WAPE" in scores
    assert "MAPE" in scores
    assert scores["MAE"] >= 0.0


def test_timesfm_es_forecaster_predict_and_evaluate():
    """Verify ES forecaster for 1h resolution streaming metrics."""
    dates = pd.date_range("2026-01-01", periods=48, freq="1h")
    df = pd.DataFrame(
        {
            "ds": dates,
            "y": np.clip(np.sin(np.linspace(0, 8, 48)) * 10.0 + 30.0, 0.0, 100.0),
        }
    )

    forecaster = ESTimesFMForecaster(resolution="1h", device="cpu", n_forecasts=6)
    forecaster.fit(df)
    fc = forecaster.predict(df, periods=6)

    assert len(fc) == 6
    assert list(fc.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert (fc["yhat"] >= 0.0).all() and (fc["yhat"] <= 100.0).all()

    eval_res = forecaster.evaluate(df, holdout_steps=6)
    assert eval_res["holdout_steps"] == 6
    assert eval_res["MAE"] >= 0.0
    assert eval_res["RMSE"] >= 0.0


def test_timesfm_es_capacity_and_plots(tmp_path):
    """Verify 30-day capacity forecast and plot generation."""
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "ds": dates,
            "y": np.clip(np.sin(np.linspace(0, 15, 100)) * 15.0 + 35.0, 0.0, 100.0),
        }
    )

    forecaster = ESTimesFMForecaster(resolution="1h", device="cpu")
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


def test_timesfm_cli_parser():
    """Verify CLI parser options for CSV and ES subcommands."""
    parser = build_tfm_parser()

    # CSV predict arguments
    args_csv = parser.parse_args(["csv", "predict", "--host", "HYDUPINTAPP16", "--periods", "14", "--device", "cpu"])
    assert args_csv.pipeline == "csv"
    assert args_csv.command == "predict"
    assert args_csv.host == "HYDUPINTAPP16"
    assert args_csv.periods == 14
    assert args_csv.device == "cpu"

    # ES evaluate arguments
    args_es = parser.parse_args(["es", "evaluate", "--resolution", "1h", "--holdout", "12", "--device", "cuda"])
    assert args_es.pipeline == "es"
    assert args_es.command == "evaluate"
    assert args_es.resolution == "1h"
    assert args_es.holdout == 12
    assert args_es.device == "cuda"
