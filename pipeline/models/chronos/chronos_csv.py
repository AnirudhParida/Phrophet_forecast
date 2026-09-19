"""
pipeline/models/chronos/chronos_csv.py
======================================
Amazon Chronos-2 universal foundation model forecaster for server host metrics
(e.g., HYDUPINTAPP16, JPRUPIWEBCRP02) across CSV and Excel telemetry.

Supports:
- Universal multi-metric forecasting with point estimates and quantile confidence bands.
- Multivariate mode (cross-learning across CPU, Memory, Disk, Network via Group Attention).
- Univariate mode (isolated independent series projection per metric).
- Auto device detection (CPU vs GPU / CUDA) with zero-code migration for GPU servers.
- Physical boundary clipping (percentage [0, 100], bytes/ops >= 0).
- Holdout evaluation (MAE, RMSE, WAPE, MAPE) and dark-themed visualization charts.
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
    CHRONOS_BATCH_SIZE,
    CHRONOS_CONTEXT_LEN,
    CHRONOS_CSV_MODELS_DIR,
    CHRONOS_DEVICE,
    CHRONOS_HOLDOUT_DAYS_CSV,
    CHRONOS_HORIZON_CSV,
    CHRONOS_MODEL_ID,
    FORECASTS_CHRONOS_CSV_DIR,
    FREQ,
    METRIC_LABELS,
    METRICS,
)
from pipeline.common.evaluation import compute_metrics
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


def _clip_metric(values: np.ndarray, metric_name: str) -> np.ndarray:
    """Clip forecasted values to physically realistic server boundaries."""
    arr = np.asarray(values, dtype=float)
    if metric_name.endswith("_pct") or "percentage" in metric_name.lower():
        return np.clip(arr, 0.0, 100.0)
    elif any(kw in metric_name.lower() for kw in ("bytes", "ops", "read", "write", "count")):
        return np.maximum(arr, 0.0)
    return arr


class ChronosCSVForecaster:
    """
    Amazon Chronos-2 universal foundation model forecaster for infrastructure host metrics.
    """

    def __init__(
        self,
        model_id: str = CHRONOS_MODEL_ID,
        device: str = CHRONOS_DEVICE,
        n_forecasts: int = CHRONOS_HORIZON_CSV,
        freq: str = FREQ,
        context_len: int = CHRONOS_CONTEXT_LEN,
        batch_size: int = CHRONOS_BATCH_SIZE,
        multivariate: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.n_forecasts = n_forecasts
        self.freq = freq
        self.context_len = context_len
        self.batch_size = batch_size
        self.multivariate = multivariate
        self._pipeline: Any = None
        self.fitted_metadata: dict[str, Any] = {}

    @property
    def pipeline(self) -> Any:
        """Lazy-load or retrieve the singleton Chronos-2 pipeline instance."""
        if self._pipeline is None:
            self._pipeline = get_chronos_pipeline(
                model_id=self.model_id,
                device=self.device,
            )
        return self._pipeline

    def fit(self, df: pd.DataFrame, host_alias: str = "host") -> None:
        """
        Validate data schema and record baseline metadata for host metrics.
        Chronos-2 is a pretrained zero-shot foundation model, so fit validates data,
        verifies context length, and registers baseline metadata.
        """
        metrics = [m for m in METRICS if m in df.columns]
        logger.info(
            "Initializing Chronos-2 (%s, multivariate=%s) on %s (%d rows, metrics: %s)...",
            self.device,
            self.multivariate,
            host_alias,
            len(df),
            metrics,
        )
        self.fitted_metadata[host_alias] = {
            "model_id": self.model_id,
            "device": self.device,
            "multivariate": self.multivariate,
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
        multivariate: Optional[bool] = None,
        host_alias: str = "host",
    ) -> dict[str, pd.DataFrame]:
        """
        Generate zero-shot multi-step future forecasts for all server metrics.

        Args:
            df: Historical DataFrame containing 'ds' and metric columns.
            periods: Forecast horizon steps ahead (default: self.n_forecasts).
            metrics: Optional list of metrics to forecast (default: METRICS in df).
            multivariate: Override whether to use multivariate cross-learning.
            host_alias: Identifier for the host.

        Returns:
            dict mapping metric_name -> DataFrame with columns:
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        """
        horizon = periods or self.n_forecasts
        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        use_multivariate = self.multivariate if multivariate is None else multivariate
        forecasts: dict[str, pd.DataFrame] = {}

        if not target_metrics:
            logger.warning("No valid metrics found in DataFrame to forecast.")
            return forecasts

        # Future date range
        last_date = pd.to_datetime(df["ds"].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1 if self.freq.upper() == "D" else 0),
            periods=horizon,
            freq=self.freq,
        )

        # Build clean input DataFrame
        input_df = df[["ds"] + target_metrics].copy()
        input_df["ds"] = pd.to_datetime(input_df["ds"])
        input_df = input_df.sort_values("ds").reset_index(drop=True)

        # Truncate to context_len if needed
        if len(input_df) > self.context_len:
            input_df = input_df.iloc[-self.context_len:].copy()

        input_df["item_id"] = host_alias

        # Predict using Chronos2Pipeline
        logger.info(
            "Running Chronos-2 inference on %s for %d metrics (multivariate=%s, horizon=%d)...",
            host_alias,
            len(target_metrics),
            use_multivariate,
            horizon,
        )

        pred_df = self.pipeline.predict_df(
            df=input_df,
            id_column="item_id",
            timestamp_column="ds",
            target=target_metrics,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.batch_size,
            cross_learning=use_multivariate,
        )

        # Parse output for each metric
        time_col = "ds" if "ds" in pred_df.columns else "timestamp"
        for metric in target_metrics:
            metric_pred = pred_df[pred_df["target_name"] == metric].sort_values(time_col)
            if metric_pred.empty:
                logger.warning("No predictions returned for metric: %s", metric)
                continue

            yhat = _clip_metric(metric_pred["0.5"].values, metric)
            yhat_lower = _clip_metric(metric_pred["0.1"].values, metric)
            yhat_upper = _clip_metric(metric_pred["0.9"].values, metric)

            forecast_dates = pd.to_datetime(metric_pred[time_col].values)
            if len(forecast_dates) != horizon:
                forecast_dates = future_dates[:len(yhat)]

            forecasts[metric] = pd.DataFrame({
                "ds": forecast_dates,
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
            })

        return forecasts

    def evaluate(
        self,
        df: pd.DataFrame,
        test_days: int = CHRONOS_HOLDOUT_DAYS_CSV,
        metrics: Optional[list[str]] = None,
        multivariate: Optional[bool] = None,
        host_alias: str = "host",
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate Chronos-2 on the most recent holdout window of data.

        Returns:
            dict mapping metric_name -> {mae, rmse, wape, mape, ...}
        """
        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        if len(df) <= test_days:
            raise ValueError(
                f"DataFrame rows ({len(df)}) must exceed holdout test_days ({test_days})."
            )

        train_df = df.iloc[:-test_days].copy()
        test_df = df.iloc[-test_days:].copy()

        logger.info(
            "Evaluating Chronos-2 holdout (%d days) on %s...",
            test_days,
            host_alias,
        )

        preds = self.predict(
            train_df,
            periods=test_days,
            metrics=target_metrics,
            multivariate=multivariate,
            host_alias=host_alias,
        )

        results: dict[str, dict[str, float]] = {}
        for metric in target_metrics:
            if metric not in preds:
                continue
            actual = test_df[metric].values
            predicted = preds[metric]["yhat"].values
            n_eval = min(len(actual), len(predicted))
            metrics_dict = compute_metrics(actual[:n_eval], predicted[:n_eval], metric_name=metric)
            # Ensure both uppercase and lowercase access
            normalized = {
                "mae": metrics_dict.get("MAE", 0.0),
                "rmse": metrics_dict.get("RMSE", 0.0),
                "wape": metrics_dict.get("WAPE", 0.0),
                "mape": metrics_dict.get("MAPE", 0.0),
                "MAE": metrics_dict.get("MAE", 0.0),
                "RMSE": metrics_dict.get("RMSE", 0.0),
                "WAPE": metrics_dict.get("WAPE", 0.0),
                "MAPE": metrics_dict.get("MAPE", 0.0),
            }
            results[metric] = normalized
            logger.info(
                "[%s] %s holdout evaluation -> MAE: %.4f, RMSE: %.4f, WAPE: %.2f%%, MAPE: %.2f%%",
                host_alias,
                metric,
                normalized["mae"],
                normalized["rmse"],
                normalized["wape"],
                normalized["mape"],
            )

        return results

    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecasts: dict[str, pd.DataFrame],
        host_alias: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate dark-themed multi-panel visualization of historical telemetry + Chronos-2 forecast.
        """
        out_dir = output_dir or FORECASTS_CHRONOS_CSV_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{host_alias}_chronos_forecast.png"

        valid_metrics = [m for m in forecasts if m in df.columns]
        n_metrics = len(valid_metrics)
        if n_metrics == 0:
            logger.warning("No metrics to plot for %s", host_alias)
            return img_path

        ncols = 2 if n_metrics > 1 else 1
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(14, 4 * nrows),
            facecolor=_PALETTE["bg"],
            squeeze=False,
        )

        mode_str = "Multivariate (Group Attention)" if self.multivariate else "Univariate"

        fig.suptitle(
            f"Amazon Chronos-2 Forecast ({mode_str}) — Host: {host_alias}",
            color=_PALETTE["text_fg"],
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )

        for idx, metric in enumerate(valid_metrics):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            ax.set_facecolor(_PALETTE["card"])

            label = METRIC_LABELS.get(metric, metric)
            f_df = forecasts[metric]

            # Plot actual historical series (last 90 days for visual clarity)
            hist_subset = df[["ds", metric]].dropna()
            if len(hist_subset) > 120:
                hist_subset = hist_subset.iloc[-120:]

            hist_ds = pd.to_datetime(hist_subset["ds"])
            ax.plot(
                hist_ds,
                hist_subset[metric],
                label="Historical Telemetry",
                color=_PALETTE["actual"],
                linewidth=1.8,
                alpha=0.9,
            )

            # Plot Chronos-2 Forecast
            f_ds = pd.to_datetime(f_df["ds"])
            ax.plot(
                f_ds,
                f_df["yhat"],
                label="Chronos-2 Forecast (p50)",
                color=_PALETTE["forecast"],
                linewidth=2.2,
            )

            # Confidence interval ribbon
            ax.fill_between(
                f_ds,
                f_df["yhat_lower"],
                f_df["yhat_upper"],
                color=_PALETTE["interval"],
                alpha=0.25,
                label="90% Confidence Interval (p10–p90)",
            )

            # Connect last historical point to first forecast point
            if not hist_subset.empty and not f_df.empty:
                ax.plot(
                    [hist_ds.iloc[-1], f_ds.iloc[0]],
                    [hist_subset[metric].iloc[-1], f_df["yhat"].iloc[0]],
                    color=_PALETTE["forecast"],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                )

            # Capacity / Warning thresholds for percentage metrics
            if metric.endswith("_pct") or "percentage" in metric.lower():
                ax.axhline(80.0, color="#E67E22", linestyle="--", linewidth=1.0, alpha=0.6, label="Warning (80%)")
                ax.axhline(90.0, color="#E74C3C", linestyle="--", linewidth=1.0, alpha=0.6, label="Critical (90%)")
                ax.set_ylim(-2, 105)

            ax.set_title(label, color=_PALETTE["text_fg"], fontsize=11, fontweight="bold", pad=8)
            ax.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.5)
            ax.tick_params(colors=_PALETTE["boundary"], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(_PALETTE["boundary"])
                spine.set_alpha(0.3)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            if idx == 0:
                ax.legend(
                    loc="upper left",
                    facecolor=_PALETTE["card"],
                    edgecolor=_PALETTE["boundary"],
                    labelcolor=_PALETTE["text_fg"],
                    fontsize=8,
                )

        # Hide empty unused subplots
        for idx in range(n_metrics, nrows * ncols):
            r, c = divmod(idx, ncols)
            fig.delaxes(axes[r][c])

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(img_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Chronos-2 forecast plot: %s", img_path)
        return img_path

    def save_forecast(
        self,
        forecasts: dict[str, pd.DataFrame],
        host_alias: str,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        Save multi-metric forecasts to standard CSV and Excel formats.
        """
        out_dir = output_dir or FORECASTS_CHRONOS_CSV_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{host_alias}_chronos_forecast.csv"
        xlsx_path = out_dir / f"{host_alias}_chronos_forecast.xlsx"

        # Combine into wide table: ds, metric_yhat, metric_lower, metric_upper
        base_df: Optional[pd.DataFrame] = None
        for metric, f_df in forecasts.items():
            renamed = f_df.rename(
                columns={
                    "yhat": f"{metric}_yhat",
                    "yhat_lower": f"{metric}_yhat_lower",
                    "yhat_upper": f"{metric}_yhat_upper",
                }
            )
            if base_df is None:
                base_df = renamed
            else:
                base_df = base_df.merge(renamed, on="ds", how="outer")

        if base_df is not None:
            base_df["host"] = host_alias
            base_df.to_csv(csv_path, index=False)
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                base_df.to_excel(writer, sheet_name="Forecast", index=False)
                # Also save individual sheets per metric
                for metric, f_df in forecasts.items():
                    f_df.to_excel(writer, sheet_name=metric[:31], index=False)

            logger.info("Saved Chronos-2 CSV forecast: %s", csv_path)
            logger.info("Saved Chronos-2 Excel forecast: %s", xlsx_path)

        return csv_path, xlsx_path
