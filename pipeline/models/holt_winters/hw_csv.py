"""
pipeline/models/holt_winters/hw_csv.py
======================================
Holt-Winters Exponential Smoothing forecaster for server host metrics
(e.g., HYDUPINTAPP16, JPRUPIWEBCRP02).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config.settings import (
    FORECASTS_HW_CSV_DIR,
    FREQ,
    HOLDOUT_DAYS,
    HW_CSV_MODELS_DIR,
    HW_DAMPED_TREND,
    HW_FORECASTS_DIR,
    HW_MODELS_DIR,
    HW_SEASONAL,
    HW_SEASONAL_PERIODS_DAILY,
    HW_TREND,
    METRIC_LABELS,
    METRICS,
    N_FORECASTS,
)
from pipeline.common.evaluation import compute_metrics

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
    """Clip forecasted values to realistic physical boundaries."""
    if metric_name.endswith("_pct") or "percentage" in metric_name.lower():
        return np.clip(values, 0.0, 100.0)
    elif "bytes" in metric_name.lower() or "ops" in metric_name.lower() or "read" in metric_name.lower() or "write" in metric_name.lower():
        return np.maximum(values, 0.0)
    return values


class HoltWintersForecaster:
    """
    Holt-Winters Exponential Smoothing forecaster for server host metrics.
    """

    def __init__(
        self,
        trend: Optional[str] = HW_TREND,
        damped_trend: bool = HW_DAMPED_TREND,
        seasonal: Optional[str] = HW_SEASONAL,
        seasonal_periods: int = HW_SEASONAL_PERIODS_DAILY,
        n_forecasts: int = N_FORECASTS,
        freq: str = FREQ,
    ) -> None:
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.n_forecasts = n_forecasts
        self.freq = freq
        self.fitted_models: dict[str, Any] = {}
        self.residual_stds: dict[str, float] = {}

    def fit_single_metric(self, series: pd.Series, metric_name: str) -> Any:
        """Fit an ExponentialSmoothing model to a single 1-D metric series."""
        n_obs = len(series)
        use_seasonal = self.seasonal

        if use_seasonal and n_obs < 2 * self.seasonal_periods:
            logger.warning(
                "[%s] Insufficient observations (%d) for seasonal_periods=%d. Falling back to seasonal=None.",
                metric_name, n_obs, self.seasonal_periods,
            )
            use_seasonal = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    series.values,
                    trend=self.trend,
                    damped_trend=self.damped_trend if self.trend else False,
                    seasonal=use_seasonal,
                    seasonal_periods=self.seasonal_periods if use_seasonal else None,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True, use_brute=False)
            except Exception as e:
                logger.warning("[%s] Estimation failed with trend='%s', seasonal='%s': %s. Retrying with fallback...",
                               metric_name, self.trend, use_seasonal, e)
                model = ExponentialSmoothing(
                    series.values,
                    trend="add",
                    damped_trend=True,
                    seasonal=None,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True, use_brute=False)

        resid = series.values - fitted.fittedvalues
        resid_std = float(np.std(resid)) if len(resid) > 0 else 1.0
        self.residual_stds[metric_name] = resid_std
        return fitted

    def fit(self, df: pd.DataFrame, metrics: Optional[list[str]] = None) -> None:
        """Fit independent Holt-Winters models for each metric."""
        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        logger.info("Fitting Holt-Winters models on %d rows for metrics: %s", len(df), target_metrics)

        for metric in target_metrics:
            logger.info("  Fitting Holt-Winters: %s (trend=%s, seasonal=%s, period=%d)",
                        metric, self.trend, self.seasonal, self.seasonal_periods)
            fitted = self.fit_single_metric(df[metric], metric)
            self.fitted_models[metric] = fitted

    def predict(
        self,
        df: pd.DataFrame,
        periods: Optional[int] = None,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate future forecasts starting from the day after the last date."""
        h = periods or self.n_forecasts
        target_metrics = metrics or list(self.fitted_models.keys()) or [m for m in METRICS if m in df.columns]
        last_date = pd.to_datetime(df["ds"].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=h,
            freq=self.freq,
        )

        forecasts: dict[str, pd.DataFrame] = {}

        for metric in target_metrics:
            if metric not in self.fitted_models:
                logger.info("Metric '%s' not fitted yet. Fitting now on provided data...", metric)
                self.fitted_models[metric] = self.fit_single_metric(df[metric], metric)

            model = self.fitted_models[metric]
            raw_pred = model.forecast(h)
            clipped_pred = _clip_metric(raw_pred, metric)

            sigma = self.residual_stds.get(metric, 1.0)
            horizon_scale = np.sqrt(1.0 + 0.05 * np.arange(h))
            ci_half_width = 1.96 * sigma * horizon_scale

            lower = _clip_metric(clipped_pred - ci_half_width, metric)
            upper = _clip_metric(clipped_pred + ci_half_width, metric)

            fc_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": clipped_pred,
                "yhat_lower": lower,
                "yhat_upper": upper,
            })
            forecasts[metric] = fc_df

        return forecasts

    def evaluate(
        self,
        df: pd.DataFrame,
        holdout_days: int = HOLDOUT_DAYS,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate Holt-Winters accuracy on trailing holdout window."""
        if holdout_days >= len(df):
            raise ValueError(f"holdout_days={holdout_days} >= total rows={len(df)}")

        target_metrics = metrics or [m for m in METRICS if m in df.columns]
        train_df = df.iloc[:-holdout_days].copy()
        test_df = df.iloc[-holdout_days:].copy()

        logger.info("Evaluating Holt-Winters: train rows=%d, holdout rows=%d...", len(train_df), len(test_df))

        results: dict[str, dict[str, float]] = {}

        for metric in target_metrics:
            eval_model = self.fit_single_metric(train_df[metric], metric)
            raw_pred = eval_model.forecast(holdout_days)
            y_pred = _clip_metric(raw_pred, metric)
            y_true = test_df[metric].values

            results[metric] = compute_metrics(y_true, y_pred, f"HW-{metric}")

        return results

    def save(self, dir_path: Path = HW_CSV_MODELS_DIR, host_alias: str = "host") -> list[Path]:
        """Save fitted Holt-Winters models to disk using joblib."""
        dir_path.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for metric, model in self.fitted_models.items():
            file_path = dir_path / f"{host_alias}_{metric}_hw.joblib"
            joblib.dump({"model": model, "resid_std": self.residual_stds.get(metric, 1.0)}, file_path)
            saved_paths.append(file_path)
            logger.info("Saved HW model: %s", file_path)
        return saved_paths

    def load(self, dir_path: Path = HW_CSV_MODELS_DIR, host_alias: str = "host", metrics: Optional[list[str]] = None) -> None:
        """Load fitted models from disk with fallback check to legacy models dir."""
        target_metrics = metrics or METRICS
        for metric in target_metrics:
            file_path = dir_path / f"{host_alias}_{metric}_hw.joblib"
            legacy_path = HW_MODELS_DIR / f"{host_alias}_{metric}_hw.joblib"

            load_path = file_path if file_path.exists() else (legacy_path if legacy_path.exists() else None)
            if load_path and load_path.exists():
                data = joblib.load(load_path)
                self.fitted_models[metric] = data["model"]
                self.residual_stds[metric] = data.get("resid_std", 1.0)
                logger.info("Loaded HW model: %s", load_path)
            else:
                logger.debug("No HW model checkpoint found for %s at %s", metric, file_path)

    def plot_forecast(
        self,
        host_alias: str,
        metric: str,
        actuals_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate forecast chart for host metric."""
        label = METRIC_LABELS.get(metric, metric)
        title = f"{host_alias} — {label} (Holt-Winters ES)"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#13131A")
            ax.set_facecolor("#1C1C28")

            ax.plot(
                actuals_df["ds"],
                actuals_df[metric],
                color=_PALETTE["actual"],
                linewidth=1.4,
                label="Historical actuals",
                alpha=0.9,
            )

            ax.plot(
                forecast_df["ds"],
                forecast_df["yhat"],
                color=_PALETTE["forecast"],
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"{len(forecast_df)}-day HW forecast",
            )

            if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
                ax.fill_between(
                    forecast_df["ds"],
                    forecast_df["yhat_lower"],
                    forecast_df["yhat_upper"],
                    color=_PALETTE["interval"],
                    alpha=0.2,
                    label="95% prediction interval",
                )

            split_date = actuals_df["ds"].iloc[-1]
            ax.axvline(x=split_date, color=_PALETTE["boundary"], linestyle=":", linewidth=1.2, alpha=0.7)

            ax.set_title(title, fontsize=13, fontweight="bold", color="#FFFFFF", pad=12)
            ax.set_xlabel("Date", fontsize=10, color="#AAAAAA")
            ax.set_ylabel(label, fontsize=10, color="#AAAAAA")
            ax.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", framealpha=0.3, facecolor=_PALETTE["text_box"])

            fig.autofmt_xdate()

            if output_file is None:
                FORECASTS_HW_CSV_DIR.mkdir(parents=True, exist_ok=True)
                output_file = FORECASTS_HW_CSV_DIR / f"forecast_hw_{host_alias}_{metric}.png"

            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved HW forecast chart: %s", output_file)
        return output_file
