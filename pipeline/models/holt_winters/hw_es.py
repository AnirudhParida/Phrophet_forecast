"""
pipeline/models/holt_winters/hw_es.py
=====================================
Holt-Winters Exponential Smoothing forecaster for Elasticsearch infrastructure metrics.
Supports minute-level, hourly, and daily aggregations, plus 30-day capacity planning.
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config.settings import (
    FORECASTS_HW_ES_DIR,
    FREQ_ES,
    FREQ_ES_DAILY,
    FREQ_ES_HOURLY,
    HOLDOUT_DAYS_ES,
    HOLDOUT_HOURS_ES,
    HOLDOUT_MINUTES_ES,
    HW_DAMPED_TREND,
    HW_ES_FORECASTS_DIR,
    HW_ES_MODELS_DIR,
    HW_SEASONAL,
    HW_SEASONAL_PERIODS_DAILY,
    HW_SEASONAL_PERIODS_HOURLY,
    HW_SEASONAL_PERIODS_MINUTELY,
    HW_TREND,
    N_FORECASTS_ES,
    N_FORECASTS_ES_DAILY,
    N_FORECASTS_ES_HOURLY,
)
from pipeline.common.evaluation import compute_metrics, mae, mape, rmse, wape

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


class ESHoltWintersForecaster:
    """
    Holt-Winters Exponential Smoothing forecaster for Elasticsearch metrics.
    """

    def __init__(
        self,
        resolution: str = "1min",
        n_forecasts: Optional[int] = None,
        seasonal_periods: Optional[int] = None,
        trend: Optional[str] = HW_TREND,
        damped_trend: bool = HW_DAMPED_TREND,
        seasonal: Optional[str] = HW_SEASONAL,
        freq: Optional[str] = None,
    ) -> None:
        self.resolution = resolution
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal

        if resolution == "1min":
            self.freq = freq or FREQ_ES
            self.n_forecasts = n_forecasts or N_FORECASTS_ES
            self.seasonal_periods = seasonal_periods or HW_SEASONAL_PERIODS_MINUTELY
            self.default_holdout = HOLDOUT_MINUTES_ES
        elif resolution == "1h":
            self.freq = freq or FREQ_ES_HOURLY
            self.n_forecasts = n_forecasts or N_FORECASTS_ES_HOURLY
            self.seasonal_periods = seasonal_periods or HW_SEASONAL_PERIODS_HOURLY
            self.default_holdout = HOLDOUT_HOURS_ES
        elif resolution == "D":
            self.freq = freq or FREQ_ES_DAILY
            self.n_forecasts = n_forecasts or N_FORECASTS_ES_DAILY
            self.seasonal_periods = seasonal_periods or HW_SEASONAL_PERIODS_DAILY
            self.default_holdout = HOLDOUT_DAYS_ES
        else:
            self.freq = freq or "1min"
            self.n_forecasts = n_forecasts or 60
            self.seasonal_periods = seasonal_periods or 60
            self.default_holdout = 60

        self.fitted_model: Optional[Any] = None
        self.resid_std: float = 1.0

    def fit(self, df: pd.DataFrame) -> None:
        """Fit Holt-Winters on ES metric series ('ds', 'y')."""
        n_obs = len(df)
        use_seasonal = self.seasonal

        if use_seasonal and n_obs < 2 * self.seasonal_periods:
            logger.warning("ES dataset length (%d) < 2 * seasonal_periods (%d). Disabling seasonality.",
                           n_obs, self.seasonal_periods)
            use_seasonal = None

        logger.info("Fitting ES Holt-Winters model: %d rows (res='%s', trend=%s, seasonal=%s, period=%d)",
                    n_obs, self.resolution, self.trend, use_seasonal, self.seasonal_periods)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    df["y"].values,
                    trend=self.trend,
                    damped_trend=self.damped_trend if self.trend else False,
                    seasonal=use_seasonal,
                    seasonal_periods=self.seasonal_periods if use_seasonal else None,
                    initialization_method="estimated",
                )
                self.fitted_model = model.fit(optimized=True, use_brute=False)
            except Exception as e:
                logger.warning("Primary HW fit failed: %s. Falling back to non-seasonal...", e)
                model = ExponentialSmoothing(
                    df["y"].values,
                    trend="add",
                    damped_trend=True,
                    seasonal=None,
                    initialization_method="estimated",
                )
                self.fitted_model = model.fit(optimized=True, use_brute=False)

        resid = df["y"].values - self.fitted_model.fittedvalues
        self.resid_std = float(np.std(resid)) if len(resid) > 0 else 1.0

    def predict(self, df: pd.DataFrame, periods: Optional[int] = None) -> pd.DataFrame:
        """Generate future forecast DataFrame for the requested horizon."""
        h = periods or self.n_forecasts
        if self.fitted_model is None:
            self.fit(df)

        last_dt = pd.to_datetime(df["ds"].iloc[-1])
        step_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(self.freq).nanos, unit="ns")
        future_dates = pd.date_range(
            start=last_dt + step_delta,
            periods=h,
            freq=self.freq,
        )

        raw_pred = self.fitted_model.forecast(h)
        yhat = np.clip(raw_pred, 0.0, 100.0)

        horizon_scale = np.sqrt(1.0 + 0.05 * np.arange(h))
        ci = 1.96 * self.resid_std * horizon_scale
        lower = np.clip(yhat - ci, 0.0, 100.0)
        upper = np.clip(yhat + ci, 0.0, 100.0)

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

    def evaluate(self, df: pd.DataFrame, holdout_steps: Optional[int] = None) -> dict[str, float]:
        """Evaluate accuracy on trailing holdout period."""
        steps = holdout_steps or self.default_holdout
        if steps >= len(df):
            raise ValueError(f"holdout_steps={steps} >= total rows={len(df)}")

        train_df = df.iloc[:-steps].copy()
        test_df = df.iloc[-steps:].copy()

        logger.info("Evaluating ES Holt-Winters: train rows=%d, holdout rows=%d...", len(train_df), len(test_df))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            use_seasonal = self.seasonal if len(train_df) >= 2 * self.seasonal_periods else None
            try:
                model = ExponentialSmoothing(
                    train_df["y"].values,
                    trend=self.trend,
                    damped_trend=self.damped_trend if self.trend else False,
                    seasonal=use_seasonal,
                    seasonal_periods=self.seasonal_periods if use_seasonal else None,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True, use_brute=False)
            except Exception:
                model = ExponentialSmoothing(train_df["y"].values, trend="add", damped_trend=True, seasonal=None)
                fitted = model.fit(optimized=True, use_brute=False)

        raw_pred = fitted.forecast(steps)
        y_pred = np.clip(raw_pred, 0.0, 100.0)
        y_true = test_df["y"].values

        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        wape_val = wape(y_true, y_pred)
        mape_val = mape(y_true, y_pred)

        print("\n" + "=" * 60)
        print(f" HOLT-WINTERS EVALUATION RESULTS ({steps} Steps — Resolution: {self.resolution})")
        print("=" * 60)
        print(f" MAE   : {mae_val:.4f} pp")
        print(f" RMSE  : {rmse_val:.4f} pp")
        print(f" WAPE  : {wape_val:.2f}%")
        print(f" MAPE  : {mape_val:.2f}%")
        print("=" * 60 + "\n")

        return {
            "holdout_steps": steps,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "WAPE": wape_val,
            "MAPE": mape_val,
        }

    def predict_capacity(self, df: pd.DataFrame, forecast_days: int = 30) -> dict[str, pd.DataFrame]:
        """Generate 30-day capacity forecast aggregated to hourly and daily predictions."""
        logger.info("Running Holt-Winters 30-day capacity forecast (%d days)...", forecast_days)
        total_minutes = forecast_days * 24 * 60

        fut_1min = self.predict(df, periods=total_minutes)

        fut_hourly = (
            fut_1min.set_index("ds")
            .resample("1h")
            .agg(
                cpu_forecast_mean=("yhat", "mean"),
                cpu_forecast_min=("yhat", "min"),
                cpu_forecast_max=("yhat", "max"),
                cpu_lower=("yhat_lower", "mean"),
                cpu_upper=("yhat_upper", "mean"),
            )
            .reset_index()
        )

        fut_daily = (
            fut_1min.set_index("ds")
            .resample("D")
            .agg(
                cpu_forecast_mean=("yhat", "mean"),
                cpu_forecast_min=("yhat", "min"),
                cpu_forecast_max=("yhat", "max"),
                cpu_lower=("yhat_lower", "mean"),
                cpu_upper=("yhat_upper", "mean"),
            )
            .reset_index()
        )

        return {
            "minutely": fut_1min,
            "hourly": fut_hourly,
            "daily": fut_daily,
        }

    def save(self, model_path: Path) -> None:
        """Save fitted model to disk using joblib."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.fitted_model, "resid_std": self.resid_std}, model_path)
        logger.info("Saved ES HW model: %s", model_path)

    def load(self, model_path: Path) -> None:
        """Load fitted model from disk."""
        data = joblib.load(model_path)
        self.fitted_model = data["model"]
        self.resid_std = data.get("resid_std", 1.0)
        logger.info("Loaded ES HW model: %s", model_path)

    def plot_forecast(
        self,
        actuals_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        entity_id: str,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Plot historical actuals and future forecast for ES metric."""
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("#13131A")
            ax.set_facecolor("#1C1C28")

            plot_actuals = actuals_df.tail(min(len(actuals_df), 1440))
            ax.plot(
                plot_actuals["ds"],
                plot_actuals["y"],
                color=_PALETTE["actual"],
                linewidth=1.2,
                label="Historical actuals",
                alpha=0.9,
            )

            ax.plot(
                forecast_df["ds"],
                forecast_df["yhat"],
                color=_PALETTE["forecast"],
                linewidth=2.0,
                linestyle="--",
                label=f"HW forecast ({self.resolution})",
            )

            if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
                ax.fill_between(
                    forecast_df["ds"],
                    forecast_df["yhat_lower"],
                    forecast_df["yhat_upper"],
                    color=_PALETTE["interval"],
                    alpha=0.25,
                    label="95% prediction interval",
                )

            split_dt = actuals_df["ds"].iloc[-1]
            ax.axvline(x=split_dt, color=_PALETTE["boundary"], linestyle=":", linewidth=1.2, alpha=0.7)

            ax.set_title(f"Elasticsearch Forecast: {entity_id} — Holt-Winters ({self.resolution})",
                         fontsize=13, fontweight="bold", color="#FFFFFF", pad=12)
            ax.set_xlabel("Timestamp", fontsize=10, color="#AAAAAA")
            ax.set_ylabel("CPU Usage (%)", fontsize=10, color="#AAAAAA")
            ax.grid(True, color=_PALETTE["grid"], linestyle="--", alpha=0.5)
            ax.legend(loc="upper left", framealpha=0.3, facecolor=_PALETTE["text_box"])

            fig.autofmt_xdate()

            if output_file is None:
                FORECASTS_HW_ES_DIR.mkdir(parents=True, exist_ok=True)
                output_file = FORECASTS_HW_ES_DIR / f"forecast_hw_es_{self.resolution}.png"

            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved ES HW forecast chart: %s", output_file)
        return output_file
