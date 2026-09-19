"""
pipeline/models/chronos/chronos_es.py
=====================================
Amazon Chronos-2 universal foundation model forecaster for Elasticsearch metrics
(1-minute, 1-hour, or daily resolution).

Supports:
- Zero-shot multi-step forecasting with point estimates and quantile uncertainty bands.
- Multivariate mode (cross-metric learning across cpu, memory, disk, network).
- Auto device detection (CPU vs GPU / CUDA) with zero-code migration for GPU servers.
- Holdout evaluation (MAE, RMSE, WAPE, MAPE).
- 30-day capacity planning forecasts aggregated to hourly/daily views.
- Publication-quality dark-themed time-series visualization charts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import (
    CHRONOS_BATCH_SIZE,
    CHRONOS_CONTEXT_LEN,
    CHRONOS_DEVICE,
    CHRONOS_ES_MODELS_DIR,
    CHRONOS_HOLDOUT_DAYS_ES,
    CHRONOS_HOLDOUT_HOURS_ES,
    CHRONOS_HOLDOUT_MINUTES_ES,
    CHRONOS_HORIZON_ES,
    CHRONOS_HORIZON_ES_DAILY,
    CHRONOS_HORIZON_ES_HOURLY,
    CHRONOS_MODEL_ID,
    ES_ENTITY_ID,
    FORECASTS_CHRONOS_ES_DIR,
    FREQ_ES,
    FREQ_ES_DAILY,
    FREQ_ES_HOURLY,
)
from pipeline.common.evaluation import mae, mape, rmse, wape
from pipeline.models.chronos.model_loader import get_chronos_pipeline, resolve_device

logger = logging.getLogger(__name__)

_PALETTE = {
    "actual": "#5B9BD5",       # steel blue
    "forecast": "#FF9900",     # amazon chronos amber
    "interval": "#FFB84D",     # amber confidence band
    "boundary": "#E0E0E0",     # light grey
    "grid": "#333344",
    "bg": "#13131A",
    "card": "#1C1C28",
    "text_fg": "#CDD6F4",
}


def _clip_metric(values: np.ndarray, metric_name: str = "cpu") -> np.ndarray:
    """Clip forecasted values to physically realistic server boundaries."""
    arr = np.asarray(values, dtype=float)
    if any(kw in metric_name.lower() for kw in ("pct", "percentage", "cpu", "mem", "filesystem")):
        return np.clip(arr, 0.0, 100.0)
    elif any(kw in metric_name.lower() for kw in ("bytes", "ops", "read", "write", "count")):
        return np.maximum(arr, 0.0)
    return arr


class ESChronosForecaster:
    """
    Amazon Chronos-2 universal forecaster for Elasticsearch infrastructure metrics.
    """

    def __init__(
        self,
        resolution: str = "1min",
        n_forecasts: Optional[int] = None,
        freq: Optional[str] = None,
        model_id: str = CHRONOS_MODEL_ID,
        device: str = CHRONOS_DEVICE,
        context_len: int = CHRONOS_CONTEXT_LEN,
        batch_size: int = CHRONOS_BATCH_SIZE,
        multivariate: bool = True,
    ) -> None:
        self.resolution = resolution
        self.model_id = model_id
        self.device = resolve_device(device)
        self.context_len = context_len
        self.batch_size = batch_size
        self.multivariate = multivariate
        self._pipeline: Any = None
        self.metadata: dict[str, Any] = {}

        if resolution == "1min":
            self.freq = freq or FREQ_ES
            self.n_forecasts = n_forecasts or CHRONOS_HORIZON_ES
            self.default_holdout = CHRONOS_HOLDOUT_MINUTES_ES
        elif resolution == "1h":
            self.freq = freq or FREQ_ES_HOURLY
            self.n_forecasts = n_forecasts or CHRONOS_HORIZON_ES_HOURLY
            self.default_holdout = CHRONOS_HOLDOUT_HOURS_ES
        elif resolution == "D":
            self.freq = freq or FREQ_ES_DAILY
            self.n_forecasts = n_forecasts or CHRONOS_HORIZON_ES_DAILY
            self.default_holdout = CHRONOS_HOLDOUT_DAYS_ES
        else:
            self.freq = freq or "1min"
            self.n_forecasts = n_forecasts or 60
            self.default_holdout = 60

    @property
    def pipeline(self) -> Any:
        """Lazy-load or retrieve the singleton Chronos-2 pipeline instance."""
        if self._pipeline is None:
            self._pipeline = get_chronos_pipeline(
                model_id=self.model_id,
                device=self.device,
            )
        return self._pipeline

    def fit(self, df: pd.DataFrame) -> None:
        """
        Validate data schema and warm up model for ES time-series.
        """
        if "ds" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'ds' timestamp column.")

        metric_cols = [c for c in df.columns if c != "ds"]
        if not metric_cols:
            raise ValueError("Input DataFrame must contain at least one metric column.")

        logger.info(
            "Chronos-2 ES model ready: %d rows (res='%s', device='%s', multivariate=%s, metrics=%s)",
            len(df),
            self.resolution,
            self.device,
            self.multivariate,
            metric_cols,
        )
        self.metadata = {
            "model_id": self.model_id,
            "device": self.device,
            "resolution": self.resolution,
            "freq": self.freq,
            "multivariate": self.multivariate,
            "rows": len(df),
            "metrics": metric_cols,
            "min_date": str(df["ds"].min()),
            "max_date": str(df["ds"].max()),
        }

    def predict(
        self,
        df: pd.DataFrame,
        periods: Optional[int] = None,
        target_col: Optional[str] = None,
        multivariate: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Generate future forecast DataFrame for the requested horizon.

        Args:
            df: Historical DataFrame with 'ds' and metric column(s).
            periods: Steps ahead to forecast (default: self.n_forecasts).
            target_col: Specific target column to forecast (default: 'y' or first metric).
            multivariate: Override whether to use multivariate cross-learning.

        Returns:
            DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        """
        h = periods or self.n_forecasts
        use_multivariate = self.multivariate if multivariate is None else multivariate

        # Identify target metrics
        available_metrics = [c for c in df.columns if c != "ds" and pd.api.types.is_numeric_dtype(df[c])]
        if not available_metrics:
            raise ValueError("No numeric metric columns found in DataFrame.")

        if target_col and target_col in available_metrics:
            chosen_targets = [target_col]
        elif "y" in available_metrics:
            chosen_targets = ["y"]
        else:
            chosen_targets = available_metrics

        # Truncate context if longer than context_len
        input_df = df[["ds"] + chosen_targets].copy()
        input_df["ds"] = pd.to_datetime(input_df["ds"])
        input_df = input_df.sort_values("ds").reset_index(drop=True)
        if len(input_df) > self.context_len:
            input_df = input_df.iloc[-self.context_len:].copy()

        input_df["item_id"] = "es_host"

        last_dt = input_df["ds"].iloc[-1]
        step_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(self.freq).nanos, unit="ns")
        future_dates = pd.date_range(
            start=last_dt + step_delta,
            periods=h,
            freq=self.freq,
        )

        logger.info(
            "Predicting Chronos-2 ES (%d steps, res='%s', multivariate=%s)...",
            h,
            self.resolution,
            use_multivariate,
        )

        pred_df = self.pipeline.predict_df(
            df=input_df,
            id_column="item_id",
            timestamp_column="ds",
            target=chosen_targets,
            prediction_length=h,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.batch_size,
            cross_learning=use_multivariate,
        )

        # Extract the primary target
        primary_target = target_col or chosen_targets[0]
        time_col = "ds" if "ds" in pred_df.columns else "timestamp"
        sub = pred_df[pred_df["target_name"] == primary_target].sort_values(time_col)

        yhat = _clip_metric(sub["0.5"].values, primary_target)
        yhat_lower = _clip_metric(sub["0.1"].values, primary_target)
        yhat_upper = _clip_metric(sub["0.9"].values, primary_target)

        forecast_dates = pd.to_datetime(sub[time_col].values)
        if len(forecast_dates) != h:
            forecast_dates = future_dates[:len(yhat)]

        result_df = pd.DataFrame({
            "ds": forecast_dates,
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
        })
        return result_df

    def evaluate(
        self,
        df: pd.DataFrame,
        test_steps: Optional[int] = None,
        target_col: str = "y",
    ) -> dict[str, Any]:
        """
        Evaluate Chronos-2 on a holdout test split.
        """
        k = test_steps or self.default_holdout
        if len(df) <= k:
            raise ValueError(f"Not enough rows ({len(df)}) for test holdout of {k} steps.")

        train_df = df.iloc[:-k].copy()
        test_df = df.iloc[-k:].copy()

        logger.info("Evaluating Chronos-2 ES on last %d steps...", k)
        forecast_df = self.predict(train_df, periods=k, target_col=target_col)

        actual = test_df[target_col].values[:k]
        predicted = forecast_df["yhat"].values[:k]

        mae_val = float(mae(actual, predicted))
        rmse_val = float(rmse(actual, predicted))
        wape_val = float(wape(actual, predicted))
        mape_val = float(mape(actual, predicted))

        logger.info(
            "Evaluation Metrics: MAE=%.4f, RMSE=%.4f, WAPE=%.2f%%, MAPE=%.2f%%",
            mae_val,
            rmse_val,
            wape_val * 100,
            mape_val,
        )

        return {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "WAPE": wape_val,
            "MAPE": mape_val,
            "device": self.device,
            "multivariate": self.multivariate,
        }

    def predict_capacity(self, df: pd.DataFrame, forecast_days: int = 30) -> dict[str, pd.DataFrame]:
        """
        Generate 30-day capacity forecast aggregated to hourly and daily predictions.
        """
        logger.info("Running Chronos-2 30-day capacity forecast (%d days, device=%s)...", forecast_days, self.device)

        target_col = "y" if "y" in df.columns else [c for c in df.columns if c != "ds"][0]

        # For capacity forecasting, if input is 1min, resample history to hourly for fast, clean capacity projections
        if self.resolution == "1min":
            df_hourly = df.set_index("ds")[[target_col]].resample("1h").mean().dropna().reset_index()
            forecaster_hourly = ESChronosForecaster(
                resolution="1h",
                model_id=self.model_id,
                device=self.device,
                context_len=self.context_len,
                multivariate=self.multivariate,
            )
            total_hours = forecast_days * 24
            fut_hourly = forecaster_hourly.predict(df_hourly, periods=total_hours, target_col=target_col)

            fut_hourly = fut_hourly.rename(
                columns={
                    "yhat": "cpu_forecast_mean",
                    "yhat_lower": "cpu_lower",
                    "yhat_upper": "cpu_upper",
                }
            )
            fut_hourly["cpu_forecast_min"] = fut_hourly["cpu_lower"]
            fut_hourly["cpu_forecast_max"] = fut_hourly["cpu_upper"]

            fut_daily = (
                fut_hourly.set_index("ds")
                .resample("D")
                .agg(
                    cpu_forecast_mean=("cpu_forecast_mean", "mean"),
                    cpu_forecast_min=("cpu_forecast_min", "min"),
                    cpu_forecast_max=("cpu_forecast_max", "max"),
                    cpu_lower=("cpu_lower", "mean"),
                    cpu_upper=("cpu_upper", "mean"),
                )
                .reset_index()
            )

            return {
                "hourly": fut_hourly,
                "daily": fut_daily,
            }
        else:
            total_steps = forecast_days * (24 if self.resolution == "1h" else 1)
            fut = self.predict(df, periods=total_steps, target_col=target_col)
            fut_hourly = fut.rename(
                columns={
                    "yhat": "cpu_forecast_mean",
                    "yhat_lower": "cpu_lower",
                    "yhat_upper": "cpu_upper",
                }
            )
            fut_hourly["cpu_forecast_min"] = fut_hourly["cpu_lower"]
            fut_hourly["cpu_forecast_max"] = fut_hourly["cpu_upper"]

            fut_daily = (
                fut_hourly.set_index("ds")
                .resample("D")
                .agg(
                    cpu_forecast_mean=("cpu_forecast_mean", "mean"),
                    cpu_forecast_min=("cpu_forecast_min", "min"),
                    cpu_forecast_max=("cpu_forecast_max", "max"),
                    cpu_lower=("cpu_lower", "mean"),
                    cpu_upper=("cpu_upper", "mean"),
                )
                .reset_index()
            )
            return {
                "hourly": fut_hourly,
                "daily": fut_daily,
            }

    def plot_capacity_forecast(
        self,
        df: pd.DataFrame,
        capacity_forecasts: dict[str, pd.DataFrame],
        entity_id: str = ES_ENTITY_ID,
        output_dir: Optional[Path] = None,
        forecast_days: int = 30,
    ) -> dict[str, Path]:
        """
        Produce publication-quality capacity forecast charts:
          1. Hourly aggregated forecast chart for N days ahead.
          2. Daily aggregated summary chart for N days ahead.

        Returns:
            dict mapping 'hourly' -> Path, 'daily' -> Path
        """
        safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
        out_dir = output_dir or (FORECASTS_CHRONOS_ES_DIR / "capacity")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}

        fut_hourly = capacity_forecasts.get("hourly", pd.DataFrame())
        fut_daily = capacity_forecasts.get("daily", pd.DataFrame())

        target_col = "y" if "y" in df.columns else [c for c in df.columns if c != "ds"][0]

        # -------------------------------------------------------------------
        # 1. Hourly Capacity Forecast Chart
        # -------------------------------------------------------------------
        if not fut_hourly.empty:
            hourly_path = out_dir / f"capacity_chronos_hourly_{safe_entity}.png"

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
                fig.patch.set_facecolor(_PALETTE["bg"])
                ax.set_facecolor(_PALETTE["card"])

                # Context: last 96 hours of hourly history
                hist_hourly = (
                    df.set_index("ds")[target_col]
                    .resample("1h")
                    .mean()
                    .dropna()
                    .reset_index()
                    .tail(96)
                )

                if not hist_hourly.empty:
                    ax.plot(
                        hist_hourly["ds"],
                        hist_hourly[target_col],
                        label=f"Recent History (last {len(hist_hourly)}h mean)",
                        color=_PALETTE["actual"],
                        linewidth=1.6,
                        alpha=0.9,
                    )

                # Forecast mean curve
                ax.plot(
                    fut_hourly["ds"],
                    fut_hourly["cpu_forecast_mean"],
                    label=f"Chronos-2 Forecast ({forecast_days}d hourly mean)",
                    color=_PALETTE["forecast"],
                    linewidth=2.0,
                    linestyle="--",
                )

                # Min-Max / Quantile Ribbon
                lower_band = fut_hourly.get("cpu_lower", fut_hourly.get("cpu_forecast_min"))
                upper_band = fut_hourly.get("cpu_upper", fut_hourly.get("cpu_forecast_max"))
                ax.fill_between(
                    fut_hourly["ds"],
                    lower_band,
                    upper_band,
                    color=_PALETTE["interval"],
                    alpha=0.25,
                    label="Projected Range (10th-90th %ile)",
                )

                # Forecast start demarcation line
                last_hist_dt = df["ds"].max()
                ax.axvline(
                    last_hist_dt,
                    color=_PALETTE["boundary"],
                    linestyle=":",
                    linewidth=1.5,
                    label="Forecast Horizon Start",
                )

                # Capacity threshold lines
                ax.axhline(
                    70.0,
                    color="#FFB74D",
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.85,
                    label="Warning Threshold (70%)",
                )
                ax.axhline(
                    85.0,
                    color="#EF5350",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.85,
                    label="Critical Threshold (85%)",
                )

                # Statistics Card
                mean_val = float(fut_hourly["cpu_forecast_mean"].mean())
                max_val = float(fut_hourly.get("cpu_forecast_max", fut_hourly["cpu_forecast_mean"]).max())
                min_val = float(fut_hourly.get("cpu_forecast_min", fut_hourly["cpu_forecast_mean"]).min())
                headroom = max(0.0, 100.0 - max_val)

                mode_str = "Multivariate" if self.multivariate else "Univariate"
                card_text = (
                    f"Amazon Chronos-2 Capacity Forecast ({mode_str})\n"
                    f"Horizon: {forecast_days} Days ({len(fut_hourly)} Hours) | Device: {self.device.upper()}\n"
                    f"────────────────────────────────\n"
                    f"Projected Mean : {mean_val:.2f}%\n"
                    f"Projected Peak : {max_val:.2f}%\n"
                    f"Projected Min  : {min_val:.2f}%\n"
                    f"Headroom (Peak): {headroom:.2f}%\n"
                    f"Status         : {'[!] AT RISK' if max_val >= 85 else ('[*] WARNING' if max_val >= 70 else '[OK] HEALTHY')}"
                )

                ax.text(
                    0.015,
                    0.96,
                    card_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontfamily="monospace",
                    fontsize=8.5,
                    color=_PALETTE["text_fg"],
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="#13131A",
                        edgecolor=_PALETTE["boundary"],
                        alpha=0.90,
                    ),
                )

                ax.set_title(
                    f"Chronos-2 Capacity Planning (Hourly Profile) — {forecast_days}-Day Outlook\nHost Entity: {entity_id}",
                    color=_PALETTE["text_fg"],
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                )
                ax.set_xlabel("Timestamp", color=_PALETTE["boundary"], fontsize=10)
                ax.set_ylabel("Metric Usage (%)", color=_PALETTE["boundary"], fontsize=10)
                ax.set_ylim(-2, 105)
                ax.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.5)
                ax.tick_params(colors=_PALETTE["boundary"], labelsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

                ax.legend(
                    loc="upper right",
                    facecolor="#13131A",
                    edgecolor=_PALETTE["boundary"],
                    labelcolor=_PALETTE["text_fg"],
                    fontsize=8.5,
                )

                plt.tight_layout()
                plt.savefig(hourly_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved Chronos-2 hourly capacity chart: %s", hourly_path)
                output_paths["hourly"] = hourly_path

        # -------------------------------------------------------------------
        # 2. Daily Summary Bar Chart
        # -------------------------------------------------------------------
        if not fut_daily.empty:
            daily_path = out_dir / f"capacity_chronos_daily_{safe_entity}.png"

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
                fig.patch.set_facecolor(_PALETTE["bg"])
                ax.set_facecolor(_PALETTE["card"])

                x_dates = pd.to_datetime(fut_daily["ds"])
                x_labels = [d.strftime("%b %d") for d in x_dates]
                x_indices = np.arange(len(x_dates))
                bar_width = 0.38

                # Dual bars: Peak Load vs Mean Load
                bars_max = ax.bar(
                    x_indices - bar_width / 2,
                    fut_daily["cpu_forecast_max"],
                    width=bar_width,
                    label="Projected Peak (Max %)",
                    color="#FF7043",
                    alpha=0.9,
                    edgecolor="#BF360C",
                    linewidth=0.8,
                )
                bars_mean = ax.bar(
                    x_indices + bar_width / 2,
                    fut_daily["cpu_forecast_mean"],
                    width=bar_width,
                    label="Projected Average (Mean %)",
                    color=_PALETTE["forecast"],
                    alpha=0.9,
                    edgecolor="#E65100",
                    linewidth=0.8,
                )

                # Error bars indicating uncertainty
                yerr_low = np.maximum(0, fut_daily["cpu_forecast_mean"] - fut_daily.get("cpu_lower", fut_daily["cpu_forecast_mean"]))
                yerr_high = np.maximum(0, fut_daily.get("cpu_upper", fut_daily["cpu_forecast_mean"]) - fut_daily["cpu_forecast_mean"])
                ax.errorbar(
                    x_indices + bar_width / 2,
                    fut_daily["cpu_forecast_mean"],
                    yerr=[yerr_low, yerr_high],
                    fmt="none",
                    ecolor="#FFF",
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.6,
                )

                # Threshold lines
                ax.axhline(70.0, color="#FFB74D", linestyle="--", linewidth=1.2, alpha=0.8, label="Warning (70%)")
                ax.axhline(85.0, color="#EF5350", linestyle="--", linewidth=1.3, alpha=0.8, label="Critical (85%)")

                ax.set_title(
                    f"Chronos-2 Capacity Planning (Daily Aggregates) — {forecast_days}-Day Summary\nHost Entity: {entity_id}",
                    color=_PALETTE["text_fg"],
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                )
                ax.set_xlabel("Date", color=_PALETTE["boundary"], fontsize=10)
                ax.set_ylabel("Utilization (%)", color=_PALETTE["boundary"], fontsize=10)
                ax.set_ylim(0, 105)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_labels, rotation=45, ha="right", color=_PALETTE["boundary"], fontsize=8.5)
                ax.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.5, axis="y")
                ax.tick_params(colors=_PALETTE["boundary"], labelsize=9)

                ax.legend(
                    loc="upper left",
                    facecolor="#13131A",
                    edgecolor=_PALETTE["boundary"],
                    labelcolor=_PALETTE["text_fg"],
                    fontsize=8.5,
                )

                plt.tight_layout()
                plt.savefig(daily_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
                plt.close(fig)
                logger.info("Saved Chronos-2 daily capacity chart: %s", daily_path)
                output_paths["daily"] = daily_path

        return output_paths

    def save_capacity_forecast(
        self,
        capacity_forecasts: dict[str, pd.DataFrame],
        entity_id: str = ES_ENTITY_ID,
        output_dir: Optional[Path] = None,
    ) -> dict[str, tuple[Path, Path]]:
        """
        Save hourly and daily capacity tables to CSV and Excel.
        """
        safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
        out_dir = output_dir or (FORECASTS_CHRONOS_ES_DIR / "capacity")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, tuple[Path, Path]] = {}

        for view in ("hourly", "daily"):
            df_view = capacity_forecasts.get(view, pd.DataFrame())
            if df_view.empty:
                continue
            csv_path = out_dir / f"capacity_chronos_{view}_{safe_entity}.csv"
            xlsx_path = out_dir / f"capacity_chronos_{view}_{safe_entity}.xlsx"

            df_view["entity_id"] = entity_id
            df_view.to_csv(csv_path, index=False)
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_view.to_excel(writer, sheet_name=f"{view.capitalize()}Capacity", index=False)

            logger.info("Saved Chronos-2 %s capacity CSV: %s", view, csv_path)
            logger.info("Saved Chronos-2 %s capacity Excel: %s", view, xlsx_path)
            paths[view] = (csv_path, xlsx_path)

        return paths
