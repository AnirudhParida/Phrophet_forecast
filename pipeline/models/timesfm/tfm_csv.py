"""
pipeline/models/timesfm/tfm_csv.py
==================================
Google TimesFM 3.0 foundation model forecaster for server host metrics
(e.g., HYDUPINTAPP16, JPRUPIWEBCRP02) across CSV and Excel data.

Supports:
- Multi-metric zero-shot forecasting with point estimates and quantile confidence bands.
- Auto device detection (CPU vs GPU / CUDA) with seamless migration.
- Physical boundary clipping (percentage [0, 100], bytes >= 0).
- Holdout evaluation and dark-themed visualization charts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import (
    FORECASTS_TFM_CSV_DIR,
    FREQ,
    METRIC_LABELS,
    METRICS,
    N_FORECASTS,
    TFM_CSV_MODELS_DIR,
    TIMESFM_CONTEXT_LEN,
    TIMESFM_DEVICE,
    TIMESFM_HOLDOUT_DAYS_CSV,
    TIMESFM_MODEL_ID,
    TIMESFM_PER_CORE_BATCH_SIZE,
)
from pipeline.common.evaluation import compute_metrics
from pipeline.models.timesfm.model_loader import get_timesfm_model, resolve_device

logger = logging.getLogger(__name__)

_PALETTE = {
    "actual": "#5B9BD5",       # steel blue
    "forecast": "#FF8C00",     # dark orange
    "interval": "#FFA500",     # orange confidence band
    "boundary": "#E0E0E0",     # light grey
    "grid": "#444444",
    "text_box": "#1E1E2E",
    "text_fg": "#CDD6F4",
}


def _clip_metric(values: np.ndarray, metric_name: str) -> np.ndarray:
    """Clip forecasted values to physically realistic server boundaries."""
    arr = np.asarray(values, dtype=float)
    if metric_name.endswith("_pct") or "percentage" in metric_name.lower():
        return np.clip(arr, 0.0, 100.0)
    elif any(kw in metric_name.lower() for kw in ("bytes", "ops", "read", "write", "count")):
        return np.maximum(arr, 0.0)
    return arr


class TimesFMForecaster:
    """
    Google TimesFM 3.0 foundation model forecaster for infrastructure host metrics.
    """

    def __init__(
        self,
        model_id: str = TIMESFM_MODEL_ID,
        device: str = TIMESFM_DEVICE,
        n_forecasts: int = N_FORECASTS,
        freq: str = FREQ,
        context_len: int = TIMESFM_CONTEXT_LEN,
        per_core_batch_size: int = TIMESFM_PER_CORE_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.n_forecasts = n_forecasts
        self.freq = freq
        self.context_len = context_len
        self.per_core_batch_size = per_core_batch_size
        self._model: Any = None
        self.fitted_metadata: dict[str, Any] = {}

    @property
    def model(self) -> Any:
        """Lazy-load or retrieve the singleton TimesFM 3.0 model instance."""
        if self._model is None:
            self._model = get_timesfm_model(
                model_id=self.model_id,
                device=self.device,
                per_core_batch_size=self.per_core_batch_size,
            )
        return self._model

    def fit(self, df: pd.DataFrame, host_alias: str = "host") -> None:
        """
        Validate data schema and warm up model for host metrics.
        TimesFM is a pretrained zero-shot foundation model, so fit validates data,
        verifies context length, and registers baseline metadata.
        """
        metrics = [m for m in METRICS if m in df.columns]
        logger.info(
            "Warming up TimesFM 3.0 (%s) on %s (%d rows, metrics: %s)...",
            self.device,
            host_alias,
            len(df),
            metrics,
        )
        self.fitted_metadata[host_alias] = {
            "model_id": self.model_id,
            "device": self.device,
            "rows": len(df),
            "metrics": metrics,
            "min_date": str(df["ds"].min()),
            "max_date": str(df["ds"].max()),
        }

    def predict(
        self,
        df: pd.DataFrame,
        periods: Optional[int] = None,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate zero-shot multi-step future forecasts for all server metrics.

        Args:
            df: Historical DataFrame containing 'ds' and metric columns.
            periods: Forecast horizon steps ahead (default: self.n_forecasts).
            metrics: Optional list of metrics to forecast (default: METRICS in df).

        Returns:
            dict mapping metric_name -> DataFrame with columns:
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        """
        horizon = periods or self.n_forecasts
        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        forecasts: dict[str, pd.DataFrame] = {}

        if not target_metrics:
            logger.warning("No valid metrics found in DataFrame to forecast.")
            return forecasts

        # Future date range
        last_date = pd.to_datetime(df["ds"].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self.freq),
            periods=horizon,
            freq=self.freq,
        )

        for metric in target_metrics:
            series = df[metric].dropna().values.astype(np.float32)
            if len(series) == 0:
                continue

            # Crop or use context window
            context = series[-self.context_len :] if len(series) > self.context_len else series
            is_positive = metric.endswith("_pct") or "bytes" in metric.lower()

            logger.debug(
                "TimesFM predicting '%s': context=%d, horizon=%d (device=%s)",
                metric,
                len(context),
                horizon,
                self.device,
            )

            out = self.model.predict(
                context=context,
                horizon=horizon,
                return_quantiles=True,
                make_positive=is_positive,
            )

            yhat = _clip_metric(out.forecast, metric)

            if out.quantiles is not None and out.quantiles.ndim >= 2:
                # 10th percentile (index 0) and 90th percentile (index 8)
                q_lower = out.quantiles[:horizon, 0]
                q_upper = out.quantiles[:horizon, -1]
            else:
                q_lower = yhat * 0.95
                q_upper = yhat * 1.05

            yhat_lower = _clip_metric(q_lower, metric)
            yhat_upper = _clip_metric(q_upper, metric)

            # Ensure lower <= upper
            yhat_lower = np.minimum(yhat_lower, yhat)
            yhat_upper = np.maximum(yhat_upper, yhat)

            forecast_df = pd.DataFrame(
                {
                    "ds": future_dates,
                    "yhat": yhat,
                    "yhat_lower": yhat_lower,
                    "yhat_upper": yhat_upper,
                }
            )
            forecasts[metric] = forecast_df

        return forecasts

    def evaluate(
        self,
        df: pd.DataFrame,
        holdout_days: int = TIMESFM_HOLDOUT_DAYS_CSV,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate forecast accuracy on a trailing holdout test window.

        Args:
            df: Full historical DataFrame.
            holdout_days: Number of trailing time steps to hold out for evaluation.
            metrics: Metrics to evaluate.

        Returns:
            dict mapping metric_name -> dictionary of error scores (MAE, RMSE, WAPE, MAPE).
        """
        n_obs = len(df)
        if n_obs <= holdout_days + 14:
            raise ValueError(
                f"Insufficient rows ({n_obs}) for holdout_days={holdout_days} + 14 context."
            )

        train_df = df.iloc[:-holdout_days].copy()
        test_df = df.iloc[-holdout_days:].copy()

        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        results: dict[str, dict[str, float]] = {}

        for metric in target_metrics:
            series_train = train_df[metric].dropna().values.astype(np.float32)
            y_true = test_df[metric].dropna().values.astype(np.float32)
            eval_horizon = len(y_true)

            if len(series_train) == 0 or eval_horizon == 0:
                continue

            context = series_train[-self.context_len :] if len(series_train) > self.context_len else series_train
            is_positive = metric.endswith("_pct") or "bytes" in metric.lower()

            out = self.model.predict(
                context=context,
                horizon=eval_horizon,
                return_quantiles=False,
                make_positive=is_positive,
            )

            y_pred = _clip_metric(out.forecast[:eval_horizon], metric)
            results[metric] = compute_metrics(y_true, y_pred, f"TimesFM-{metric}")

        return results

    def save(
        self,
        dir_path: Path = TFM_CSV_MODELS_DIR,
        host_alias: str = "host",
    ) -> list[Path]:
        """Save model checkpoint metadata and configuration."""
        dir_path.mkdir(parents=True, exist_ok=True)
        meta_file = dir_path / f"{host_alias}_tfm_meta.json"
        meta = {
            "host": host_alias,
            "model_id": self.model_id,
            "device": self.device,
            "context_len": self.context_len,
            "n_forecasts": self.n_forecasts,
            "freq": self.freq,
            "metadata": self.fitted_metadata.get(host_alias, {}),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved TimesFM metadata: %s", meta_file)
        return [meta_file]

    def load(
        self,
        dir_path: Path = TFM_CSV_MODELS_DIR,
        host_alias: str = "host",
        metrics: Optional[list[str]] = None,
    ) -> None:
        """Load metadata for host metrics."""
        meta_file = dir_path / f"{host_alias}_tfm_meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.fitted_metadata[host_alias] = data.get("metadata", {})
            logger.info("Loaded TimesFM metadata for %s from %s", host_alias, meta_file)

    def plot_forecast(
        self,
        host_alias: str,
        metric: str,
        actuals_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate a dark-themed forecast visualization PNG chart."""
        label = METRIC_LABELS.get(metric, metric)
        title = f"{host_alias} — {label} (Google TimesFM 3.0)"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#13131A")
            ax.set_facecolor("#1C1C28")

            # Historical actuals
            ax.plot(
                actuals_df["ds"],
                actuals_df[metric],
                color=_PALETTE["actual"],
                linewidth=1.4,
                label="Historical actuals",
                alpha=0.9,
            )

            # Forecasted line
            ax.plot(
                forecast_df["ds"],
                forecast_df["yhat"],
                color=_PALETTE["forecast"],
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"{len(forecast_df)}-step TimesFM 3.0 forecast",
            )

            # 10th-90th percentile confidence band
            ax.fill_between(
                forecast_df["ds"],
                forecast_df["yhat_lower"],
                forecast_df["yhat_upper"],
                color=_PALETTE["interval"],
                alpha=0.22,
                label="80% Prediction Band (10th-90th %ile)",
            )

            # Forecast start divider line
            first_fc_ds = forecast_df["ds"].iloc[0]
            ax.axvline(
                first_fc_ds,
                color="#888888",
                linestyle=":",
                linewidth=1.2,
                label="Forecast start",
            )

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=_PALETTE["text_fg"])
            ax.set_xlabel("Date", fontsize=10, color=_PALETTE["text_fg"])
            ax.set_ylabel(label, fontsize=10, color=_PALETTE["text_fg"])
            ax.grid(True, color=_PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(loc="upper left", framealpha=0.8, facecolor="#13131A", edgecolor="#444444")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
            fig.autofmt_xdate(rotation=20)
            plt.tight_layout()

            out = output_file or (FORECASTS_TFM_CSV_DIR / f"forecast_tfm_{host_alias}_{metric}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)

        logger.info("Saved TimesFM chart: %s", out)
        return out
