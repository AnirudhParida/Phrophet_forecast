"""
pipeline/models/timesfm/tfm_es.py
=================================
Google TimesFM 3.0 foundation model forecaster for Elasticsearch metrics
(1-minute, 1-hour, or daily resolution).

Supports:
- Zero-shot multi-step forecasting with point estimates and quantile uncertainty bands.
- Auto device detection (CPU vs GPU / CUDA) with seamless migration.
- Holdout evaluation (MAE, RMSE, WAPE, MAPE).
- 30-day capacity planning forecasts aggregated to hourly/daily views.
- Publication-quality dark-themed time-series visualization charts.
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
    ES_ENTITY_ID,
    FORECASTS_TFM_ES_DIR,
    FREQ_ES,
    FREQ_ES_DAILY,
    FREQ_ES_HOURLY,
    TFM_ES_MODELS_DIR,
    TIMESFM_CONTEXT_LEN,
    TIMESFM_DEVICE,
    TIMESFM_HOLDOUT_DAYS_ES,
    TIMESFM_HOLDOUT_HOURS_ES,
    TIMESFM_HOLDOUT_MINUTES_ES,
    TIMESFM_HORIZON_ES,
    TIMESFM_HORIZON_ES_DAILY,
    TIMESFM_HORIZON_ES_HOURLY,
    TIMESFM_MODEL_ID,
    TIMESFM_PER_CORE_BATCH_SIZE,
)
from pipeline.common.evaluation import mae, mape, rmse, wape
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


class ESTimesFMForecaster:
    """
    Google TimesFM 3.0 forecaster for Elasticsearch infrastructure metrics.
    """

    def __init__(
        self,
        resolution: str = "1min",
        n_forecasts: Optional[int] = None,
        freq: Optional[str] = None,
        model_id: str = TIMESFM_MODEL_ID,
        device: str = TIMESFM_DEVICE,
        context_len: int = TIMESFM_CONTEXT_LEN,
        per_core_batch_size: int = TIMESFM_PER_CORE_BATCH_SIZE,
    ) -> None:
        self.resolution = resolution
        self.model_id = model_id
        self.device = resolve_device(device)
        self.context_len = context_len
        self.per_core_batch_size = per_core_batch_size
        self._model: Any = None
        self.metadata: dict[str, Any] = {}

        if resolution == "1min":
            self.freq = freq or FREQ_ES
            self.n_forecasts = n_forecasts or TIMESFM_HORIZON_ES
            self.default_holdout = TIMESFM_HOLDOUT_MINUTES_ES
        elif resolution == "1h":
            self.freq = freq or FREQ_ES_HOURLY
            self.n_forecasts = n_forecasts or TIMESFM_HORIZON_ES_HOURLY
            self.default_holdout = TIMESFM_HOLDOUT_HOURS_ES
        elif resolution == "D":
            self.freq = freq or FREQ_ES_DAILY
            self.n_forecasts = n_forecasts or TIMESFM_HORIZON_ES_DAILY
            self.default_holdout = TIMESFM_HOLDOUT_DAYS_ES
        else:
            self.freq = freq or "1min"
            self.n_forecasts = n_forecasts or 60
            self.default_holdout = 60

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

    def fit(self, df: pd.DataFrame) -> None:
        """
        Validate data schema and warm up model for ES time-series.
        """
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError("Input DataFrame must contain 'ds' and 'y' columns.")

        logger.info(
            "TimesFM ES model ready: %d rows (res='%s', device='%s')",
            len(df),
            self.resolution,
            self.device,
        )
        self.metadata = {
            "model_id": self.model_id,
            "device": self.device,
            "resolution": self.resolution,
            "freq": self.freq,
            "rows": len(df),
            "min_date": str(df["ds"].min()),
            "max_date": str(df["ds"].max()),
        }

    def predict(self, df: pd.DataFrame, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Generate future forecast DataFrame for the requested horizon.

        Args:
            df: Historical DataFrame with 'ds' and 'y'.
            periods: Steps ahead to forecast (default: self.n_forecasts).

        Returns:
            DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        """
        h = periods or self.n_forecasts
        series = df["y"].dropna().values.astype(np.float32)

        if len(series) == 0:
            raise ValueError("Input series 'y' is empty.")

        context = series[-self.context_len :] if len(series) > self.context_len else series

        last_dt = pd.to_datetime(df["ds"].iloc[-1])
        step_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(self.freq).nanos, unit="ns")
        future_dates = pd.date_range(
            start=last_dt + step_delta,
            periods=h,
            freq=self.freq,
        )

        logger.debug(
            "TimesFM ES predicting: context=%d, horizon=%d, res='%s' (device=%s)",
            len(context),
            h,
            self.resolution,
            self.device,
        )

        out = self.model.predict(
            context=context,
            horizon=h,
            return_quantiles=True,
            make_positive=True,
        )

        # Clip percentage values to [0, 100]
        yhat = np.clip(out.forecast[:h], 0.0, 100.0)

        if out.quantiles is not None and out.quantiles.ndim >= 2:
            q_lower = out.quantiles[:h, 0]
            q_upper = out.quantiles[:h, -1]
        else:
            q_lower = yhat * 0.95
            q_upper = yhat * 1.05

        lower = np.clip(np.minimum(q_lower, yhat), 0.0, 100.0)
        upper = np.clip(np.maximum(q_upper, yhat), 0.0, 100.0)

        return pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": lower,
                "yhat_upper": upper,
            }
        )

    def evaluate(self, df: pd.DataFrame, holdout_steps: Optional[int] = None) -> dict[str, float]:
        """
        Evaluate forecast accuracy on trailing holdout period.
        """
        steps = holdout_steps or self.default_holdout
        if steps >= len(df):
            raise ValueError(f"holdout_steps={steps} >= total rows={len(df)}")

        train_df = df.iloc[:-steps].copy()
        test_df = df.iloc[-steps:].copy()

        logger.info(
            "Evaluating ES TimesFM 3.0: train rows=%d, holdout rows=%d (device=%s)...",
            len(train_df),
            len(test_df),
            self.device,
        )

        series_train = train_df["y"].dropna().values.astype(np.float32)
        y_true = test_df["y"].dropna().values.astype(np.float32)
        eval_h = len(y_true)

        context = series_train[-self.context_len :] if len(series_train) > self.context_len else series_train

        out = self.model.predict(
            context=context,
            horizon=eval_h,
            return_quantiles=False,
            make_positive=True,
        )

        y_pred = np.clip(out.forecast[:eval_h], 0.0, 100.0)

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        wape_val = wape(y_true, y_pred)
        mape_val = mape(y_true, y_pred)

        print("\n" + "=" * 60)
        print(f" TIMESFM 3.0 ES EVALUATION RESULTS ({eval_h} Steps — Res: {self.resolution})")
        print("=" * 60)
        print(f" Target Device : {self.device.upper()}")
        print(f" MAE           : {mae_val:.4f} pp")
        print(f" RMSE          : {rmse_val:.4f} pp")
        print(f" WAPE          : {wape_val:.2f}%")
        print(f" MAPE          : {mape_val:.2f}%")
        print("=" * 60 + "\n")

        return {
            "holdout_steps": eval_h,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "WAPE": wape_val,
            "MAPE": mape_val,
            "device": self.device,
        }

    def predict_capacity(self, df: pd.DataFrame, forecast_days: int = 30) -> dict[str, pd.DataFrame]:
        """
        Generate 30-day capacity forecast aggregated to hourly and daily predictions.
        """
        logger.info("Running TimesFM 30-day capacity forecast (%d days, device=%s)...", forecast_days, self.device)

        # For capacity forecasting, if input is 1min, we can aggregate history to 1h first or forecast directly
        if self.resolution == "1min":
            # For 30 days of minute data (43,200 points), resample to hourly for fast, clean capacity projections
            df_hourly = df.set_index("ds").resample("1h").mean().dropna().reset_index()
            forecaster_hourly = ESTimesFMForecaster(
                resolution="1h",
                model_id=self.model_id,
                device=self.device,
                context_len=self.context_len,
            )
            total_hours = forecast_days * 24
            fut_hourly = forecaster_hourly.predict(df_hourly, periods=total_hours)

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
            fut = self.predict(df, periods=total_steps)
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
        import re

        safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
        out_dir = output_dir or (FORECASTS_TFM_ES_DIR / "capacity")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}

        fut_hourly = capacity_forecasts.get("hourly", pd.DataFrame())
        fut_daily = capacity_forecasts.get("daily", pd.DataFrame())

        # -------------------------------------------------------------------
        # 1. Hourly Capacity Forecast Chart
        # -------------------------------------------------------------------
        if not fut_hourly.empty:
            hourly_path = out_dir / f"capacity_tfm_hourly_{safe_entity}.png"

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
                fig.patch.set_facecolor("#13131A")
                ax.set_facecolor("#1C1C28")

                # Context: last 96 hours of hourly history
                hist_hourly = (
                    df.set_index("ds")["y"]
                    .resample("1h")
                    .mean()
                    .dropna()
                    .reset_index()
                    .tail(96)
                )

                if not hist_hourly.empty:
                    ax.plot(
                        hist_hourly["ds"],
                        hist_hourly["y"],
                        label=f"Recent History (last {len(hist_hourly)}h mean)",
                        color=_PALETTE["actual"],
                        linewidth=1.6,
                        alpha=0.9,
                    )

                # Forecast mean curve
                ax.plot(
                    fut_hourly["ds"],
                    fut_hourly["cpu_forecast_mean"],
                    label=f"TimesFM 3.0 Forecast ({forecast_days}d hourly mean)",
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
                    color="#E0E0E0",
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

                card_text = (
                    f"Google TimesFM 3.0 Capacity Forecast\n"
                    f"Horizon: {forecast_days} Days ({len(fut_hourly)} Hours) | Device: {self.device.upper()}\n"
                    f"────────────────────────────────\n"
                    f"Projected Mean : {mean_val:.2f}%\n"
                    f"Projected Peak : {max_val:.2f}%\n"
                    f"Projected Min  : {min_val:.2f}%\n"
                    f"Min Headroom   : {headroom:.2f}%"
                )
                ax.text(
                    0.015,
                    0.96,
                    card_text,
                    transform=ax.transAxes,
                    fontsize=9.5,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor=_PALETTE["text_box"],
                        edgecolor="#FF8C00",
                        alpha=0.92,
                    ),
                    color=_PALETTE["text_fg"],
                )

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, forecast_days // 10)))
                fig.autofmt_xdate(rotation=25)

                ax.set_title(
                    f"{forecast_days}-Day Infrastructure Capacity Forecast (Hourly) — {entity_id} [Google TimesFM 3.0]",
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                    color=_PALETTE["text_fg"],
                )
                ax.set_xlabel("Timeline", fontsize=11, color=_PALETTE["text_fg"])
                ax.set_ylabel("CPU Utilization (%)", fontsize=11, color=_PALETTE["text_fg"])
                ax.set_ylim(0, 105)
                ax.grid(True, color=_PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
                ax.legend(loc="upper right", framealpha=0.85, facecolor="#13131A", edgecolor="#444444", fontsize=9)

                fig.tight_layout()
                fig.savefig(hourly_path, dpi=150, facecolor=fig.get_facecolor())
                plt.close(fig)
                output_paths["hourly"] = hourly_path
                logger.info("Saved hourly capacity forecast chart: %s", hourly_path)

        # -------------------------------------------------------------------
        # 2. Daily Capacity Summary Chart
        # -------------------------------------------------------------------
        if not fut_daily.empty:
            daily_path = out_dir / f"capacity_tfm_daily_{safe_entity}.png"

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
                fig.patch.set_facecolor("#13131A")
                ax.set_facecolor("#1C1C28")

                # Recent daily history (last 7 days)
                hist_daily = (
                    df.set_index("ds")["y"]
                    .resample("D")
                    .mean()
                    .dropna()
                    .reset_index()
                    .tail(7)
                )

                if not hist_daily.empty:
                    ax.bar(
                        hist_daily["ds"],
                        hist_daily["y"],
                        width=0.6,
                        label="Historical (Daily Mean)",
                        color=_PALETTE["actual"],
                        alpha=0.75,
                        edgecolor="#4A90E2",
                    )

                # Future daily bars
                ax.bar(
                    fut_daily["ds"],
                    fut_daily["cpu_forecast_mean"],
                    width=0.6,
                    label=f"TimesFM 3.0 Forecast (Daily Mean)",
                    color=_PALETTE["forecast"],
                    alpha=0.85,
                    edgecolor="#FFB347",
                )

                # Error bars showing min to max daily spread
                y_err_lower = np.maximum(0.0, fut_daily["cpu_forecast_mean"] - fut_daily["cpu_forecast_min"])
                y_err_upper = np.maximum(0.0, fut_daily["cpu_forecast_max"] - fut_daily["cpu_forecast_mean"])
                ax.errorbar(
                    fut_daily["ds"],
                    fut_daily["cpu_forecast_mean"],
                    yerr=[y_err_lower, y_err_upper],
                    fmt="none",
                    color="#FFD54F",
                    capsize=3.5,
                    linewidth=1.3,
                    label="Daily Min–Max Peak Spread",
                )

                # Forecast start line
                ax.axvline(
                    df["ds"].max(),
                    color="#E0E0E0",
                    linestyle=":",
                    linewidth=1.5,
                    label="Forecast Horizon Start",
                )

                # Capacity thresholds
                ax.axhline(70.0, color="#FFB74D", linestyle="--", linewidth=1.3, alpha=0.85, label="Warning (70%)")
                ax.axhline(85.0, color="#EF5350", linestyle="--", linewidth=1.4, alpha=0.85, label="Critical (85%)")

                # Statistics Card
                avg_daily_mean = float(fut_daily["cpu_forecast_mean"].mean())
                peak_daily = float(fut_daily["cpu_forecast_max"].max())
                days_over_70 = int((fut_daily["cpu_forecast_max"] >= 70.0).sum())
                days_over_85 = int((fut_daily["cpu_forecast_max"] >= 85.0).sum())

                summary_text = (
                    f"30-Day Daily Capacity Summary\n"
                    f"─────────────────────────────\n"
                    f"Avg Daily Mean   : {avg_daily_mean:.2f}%\n"
                    f"Peak Single Hour : {peak_daily:.2f}%\n"
                    f"Days Peak >= 70% : {days_over_70} days\n"
                    f"Days Peak >= 85% : {days_over_85} days\n"
                    f"Status           : {'HEALTHY' if peak_daily < 70 else ('WARNING' if peak_daily < 85 else 'CRITICAL')}"
                )
                ax.text(
                    0.015,
                    0.96,
                    summary_text,
                    transform=ax.transAxes,
                    fontsize=9.5,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor=_PALETTE["text_box"],
                        edgecolor="#FF8C00",
                        alpha=0.92,
                    ),
                    color=_PALETTE["text_fg"],
                )

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, forecast_days // 10)))
                fig.autofmt_xdate(rotation=25)

                ax.set_title(
                    f"{forecast_days}-Day Capacity Planning (Daily Aggregates) — {entity_id} [Google TimesFM 3.0]",
                    fontsize=13,
                    fontweight="bold",
                    pad=14,
                    color=_PALETTE["text_fg"],
                )
                ax.set_xlabel("Timeline", fontsize=11, color=_PALETTE["text_fg"])
                ax.set_ylabel("CPU Utilization (%)", fontsize=11, color=_PALETTE["text_fg"])
                ax.set_ylim(0, 105)
                ax.grid(True, axis="y", color=_PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
                ax.legend(loc="upper right", framealpha=0.85, facecolor="#13131A", edgecolor="#444444", fontsize=9)

                fig.tight_layout()
                fig.savefig(daily_path, dpi=150, facecolor=fig.get_facecolor())
                plt.close(fig)
                output_paths["daily"] = daily_path
                logger.info("Saved daily capacity forecast chart: %s", daily_path)

        return output_paths

    def save(self, model_path: Path) -> None:
        """Save metadata and configuration to disk."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Change suffix to json if joblib passed
        meta_path = model_path.with_suffix(".json")
        data = {
            "model_id": self.model_id,
            "device": self.device,
            "resolution": self.resolution,
            "freq": self.freq,
            "metadata": self.metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved TimesFM ES metadata: %s", meta_path)

    def load(self, model_path: Path) -> None:
        """Load metadata and configuration from disk."""
        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = data.get("metadata", {})
            logger.info("Loaded TimesFM ES metadata: %s", meta_path)

    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        entity_id: str = ES_ENTITY_ID,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate forecast chart for Elasticsearch metrics."""
        title = f"TimesFM 3.0 Forecast — {entity_id} ({self.resolution} | {self.device.upper()})"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#13131A")
            ax.set_facecolor("#1C1C28")

            # Slice history for readable chart
            plot_history = df.tail(min(len(df), 500))

            ax.plot(
                plot_history["ds"],
                plot_history["y"],
                color=_PALETTE["actual"],
                linewidth=1.2,
                label=f"Historical Actuals ({len(plot_history)} obs)",
                alpha=0.85,
            )

            ax.plot(
                forecast_df["ds"],
                forecast_df["yhat"],
                color=_PALETTE["forecast"],
                linewidth=2.0,
                linestyle="--",
                marker="o" if len(forecast_df) <= 48 else None,
                markersize=3,
                label=f"{len(forecast_df)}-step TimesFM 3.0 forecast",
            )

            ax.fill_between(
                forecast_df["ds"],
                forecast_df["yhat_lower"],
                forecast_df["yhat_upper"],
                color=_PALETTE["interval"],
                alpha=0.22,
                label="80% Prediction Band (10th-90th %ile)",
            )

            first_fc_ds = forecast_df["ds"].iloc[0]
            ax.axvline(
                first_fc_ds,
                color="#888888",
                linestyle=":",
                linewidth=1.2,
                label="Forecast Start",
            )

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=_PALETTE["text_fg"])
            ax.set_xlabel("Timestamp", fontsize=10, color=_PALETTE["text_fg"])
            ax.set_ylabel("CPU Usage (%)", fontsize=10, color=_PALETTE["text_fg"])
            ax.set_ylim(0, 105)
            ax.grid(True, color=_PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(loc="upper left", framealpha=0.8, facecolor="#13131A", edgecolor="#444444")

            if self.resolution in ("1min", "1h"):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))

            fig.autofmt_xdate(rotation=20)
            plt.tight_layout()

            out = output_file or (
                FORECASTS_TFM_ES_DIR / f"forecast_tfm_es_{self.resolution}.png"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)

        logger.info("Saved TimesFM ES plot: %s", out)
        return out
