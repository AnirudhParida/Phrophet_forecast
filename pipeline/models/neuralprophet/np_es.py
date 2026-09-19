"""
pipeline/es_forecaster.py
=========================
NeuralProphet forecasting wrapper for 1-minute resolution Elasticsearch metrics.

Responsibilities
----------------
1. Train a NeuralProphet model on minute-level metric data ('ds', 'y').
2. Evaluate forecast accuracy (MAE, RMSE, MAPE) on holdout test windows.
3. Generate N-step ahead future forecasts.
4. Generate 30-day capacity planning forecasts (1-min model, aggregated to hourly/daily).
5. Save and load trained model checkpoints.
6. Create visualization charts for historical values and future predictions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralprophet import NeuralProphet

from config.settings import (
    BATCH_SIZE_ES,
    EPOCHS_ES,
    ES_FORECASTS_DIR,
    ES_MODELS_DIR,
    FREQ_ES,
    HOLDOUT_MINUTES_ES,
    LEARNING_RATE_ES,
    N_FORECASTS_ES,
    N_LAGS_ES,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def extract_es_forecast_rows(
    forecast_df: pd.DataFrame,
    n_forecasts: int = N_FORECASTS_ES,
) -> pd.DataFrame:
    """
    Extract clean future forecast rows (ds, yhat) from a multi-step NeuralProphet forecast DataFrame.
    """
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=["ds", "yhat"])

    # Select future rows (where y is NaN or taking the last n_forecasts rows)
    if "y" in forecast_df.columns and forecast_df["y"].isna().any():
        future_rows = forecast_df[forecast_df["y"].isna()].copy()
    else:
        future_rows = forecast_df.tail(n_forecasts).copy()

    yhat_cols = sorted(
        [c for c in forecast_df.columns if c.startswith("yhat") and c[4:].isdigit()],
        key=lambda c: int(c[4:]),
    )

    yhat_vals = []
    for i, (_, row) in enumerate(future_rows.iterrows(), start=1):
        col = f"yhat{i}"
        if col in row and pd.notna(row[col]):
            val = float(row[col])
        else:
            non_nulls = row[yhat_cols].dropna() if yhat_cols else pd.Series(dtype=float)
            val = float(non_nulls.iloc[0]) if not non_nulls.empty else np.nan
        yhat_vals.append({"ds": row["ds"], "yhat": val})

    return pd.DataFrame(yhat_vals)


class ESMetricsForecaster:
    """
    NeuralProphet forecaster designed for 1-minute resolution infrastructure metrics.
    """

    def __init__(
        self,
        n_lags: int = N_LAGS_ES,
        n_forecasts: int = N_FORECASTS_ES,
        epochs: int = EPOCHS_ES,
        batch_size: int = BATCH_SIZE_ES,
        learning_rate: float = LEARNING_RATE_ES,
        freq: str = FREQ_ES,
    ) -> None:
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.freq = freq
        self.model: Optional[NeuralProphet] = None

    def _build_model(self) -> NeuralProphet:
        """Instantiate a NeuralProphet model with minute-level parameters."""
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        m = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            loss_func="Huber",
            normalize="minmax",
        )
        return m

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Fit NeuralProphet model on minute-level DataFrame containing ['ds', 'y'].
        """
        logger.info(
            "Fitting NeuralProphet ES model (n_lags=%d, n_forecasts=%d, epochs=%d) on %d rows...",
            self.n_lags,
            self.n_forecasts,
            self.epochs,
            len(df),
        )
        self.model = self._build_model()
        metrics = self.model.fit(df, freq=self.freq)
        logger.info("Model fitting complete.")
        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate future predictions for specified periods ahead.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        forecast_periods = periods or self.n_forecasts
        logger.info("Generating %d-step future forecast (%s frequency)...", forecast_periods, self.freq)

        future = self.model.make_future_dataframe(
            df,
            periods=forecast_periods,
            n_historic_predictions=True,
        )
        forecast = self.model.predict(future)
        return forecast

    def evaluate(
        self,
        df: pd.DataFrame,
        holdout_minutes: int = HOLDOUT_MINUTES_ES,
    ) -> dict[str, float]:
        """
        Evaluate model accuracy by holding out the trailing N minutes.
        """
        if len(df) <= holdout_minutes + self.n_lags:
            raise ValueError(
                f"DataFrame has insufficient rows ({len(df)}) for holdout window ({holdout_minutes}) "
                f"plus n_lags ({self.n_lags})."
            )

        train_df = df.iloc[:-holdout_minutes].copy()
        test_df = df.iloc[-holdout_minutes:].copy()

        logger.info(
            "Evaluating model: train rows=%d, holdout test rows=%d...",
            len(train_df),
            len(test_df),
        )

        eval_model = self._build_model()
        eval_model.fit(train_df, freq=self.freq)

        # Make prediction across the holdout range
        future = eval_model.make_future_dataframe(
            train_df,
            periods=holdout_minutes,
            n_historic_predictions=False,
        )
        forecast = eval_model.predict(future)

        # Align actual vs predicted values
        fut_eval = extract_es_forecast_rows(forecast, n_forecasts=holdout_minutes)
        y_true = test_df["y"].values
        y_pred = fut_eval["yhat"].values[: len(y_true)]

        if len(y_pred) < len(y_true):
            y_true = y_true[: len(y_pred)]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # Avoid division by zero in MAPE
        nonzero_mask = np.abs(y_true) > 1e-5
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
        else:
            mape = 0.0

        eval_results = {
            "holdout_minutes": holdout_minutes,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }

        print("\n" + "=" * 60)
        print(f" HOLDOUT EVALUATION RESULTS ({holdout_minutes} Minutes)")
        print("=" * 60)
        print(f" MAE  : {mae:.4f}")
        print(f" RMSE : {rmse:.4f}")
        print(f" MAPE : {mape:.2f}%")
        print("=" * 60 + "\n")

        return eval_results

    def save(self, model_path: Path) -> None:
        """Save trained NeuralProphet model checkpoint to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an un-fitted model.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, model_path)
        logger.info("Saved trained ES model to checkpoint: %s", model_path)

    def load(self, model_path: Path) -> None:
        """Load trained NeuralProphet model checkpoint from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = torch.load(model_path, weights_only=False)
        except TypeError:
            self.model = torch.load(model_path)
        logger.info("Loaded ES model from checkpoint: %s", model_path)

    # -----------------------------------------------------------------------
    # 30-Day Capacity Planning Forecast
    # -----------------------------------------------------------------------

    def predict_capacity(
        self,
        df: pd.DataFrame,
        forecast_days: int = 30,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate a 30-day capacity planning forecast from 1-minute training data.

        Strategy (Trend + Seasonality — Stable Long-Horizon)
        -----------------------------------------------------
        Autoregressive models (n_lags>0) compound errors over 720 rolls and
        produce explosive results for 30-day horizons. Instead we:

        1. Resample 1-minute training data → hourly (captures daily/weekly cycles).
        2. Train a lightweight NeuralProphet with **n_lags=0** (no AR) using only
           trend + daily/weekly seasonality.  These components are well-behaved
           indefinitely into the future.
        3. Use ``make_future_dataframe(periods=forecast_days*24)`` to directly
           predict 720 hourly steps in a single, stable forward pass.
        4. Aggregate the hourly forecast to daily for the capacity table.

        Parameters
        ----------
        df           : Historical 1-minute DataFrame with ['ds', 'y'].
        forecast_days: Number of future days to forecast (default: 30).

        Returns
        -------
        dict with keys:
            ``hourly``  – pd.DataFrame with columns ['ds', 'yhat_mean', 'yhat_max', 'yhat_min'] per hour
            ``daily``   – pd.DataFrame with columns ['ds', 'yhat_mean', 'yhat_max', 'yhat_min'] per day
            ``minutely``– empty DataFrame (minute-level not applicable for long-horizon)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        total_hours = forecast_days * 24

        logger.info(
            "Generating %d-day capacity forecast using trend+seasonality model "
            "(%d hourly steps)...", forecast_days, total_hours,
        )

        # 1. Resample 1-minute training data to hourly
        df_hourly = (
            df.set_index("ds")["y"]
            .resample("1h")
            .mean()
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .reset_index()
            .rename(columns={"y": "y"})
        )
        logger.info("Resampled to %d hourly rows for capacity model.", len(df_hourly))

        # 2. Train a dedicated capacity model (no autoregression)
        #    Trend + daily seasonality + weekly seasonality only — stable for 30-day horizon
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        cap_model = NeuralProphet(
            n_lags=0,                    # NO autoregression — avoids compounding errors
            n_forecasts=1,               # single-step per training sample (standard prophet mode)
            epochs=150,
            batch_size=32,
            learning_rate=1e-3,
            daily_seasonality=True,      # captures intraday CPU cycles
            weekly_seasonality=True,     # captures weekday vs weekend patterns
            yearly_seasonality=False,    # not enough data
            trend_reg=0.5,               # mild regularisation to prevent trend runaway
            normalize="minmax",
        )
        logger.info("Fitting capacity model on %d hourly rows (no AR)...", len(df_hourly))
        cap_model.fit(df_hourly, freq="1h")

        # 3. Predict total_hours steps into the future in a single stable forward pass
        future_hourly = cap_model.make_future_dataframe(
            df_hourly,
            periods=total_hours,
            n_historic_predictions=False,
        )
        raw_forecast = cap_model.predict(future_hourly)

        # 4. Extract future rows — with n_lags=0 the yhat column is 'yhat1' (single step)
        fut_rows = raw_forecast[raw_forecast["y"].isna()].copy() if "y" in raw_forecast.columns else raw_forecast.tail(total_hours).copy()
        if "yhat1" in fut_rows.columns:
            fut_rows = fut_rows[["ds", "yhat1"]].rename(columns={"yhat1": "yhat"})
        elif "yhat" in fut_rows.columns:
            fut_rows = fut_rows[["ds", "yhat"]]
        else:
            # Fallback: find first yhat column
            yhat_col = next((c for c in fut_rows.columns if c.startswith("yhat")), None)
            if yhat_col:
                fut_rows = fut_rows[["ds", yhat_col]].rename(columns={yhat_col: "yhat"})
            else:
                logger.error("No yhat column found in capacity forecast output.")
                return {"minutely": pd.DataFrame(), "hourly": pd.DataFrame(), "daily": pd.DataFrame()}

        fut_rows = fut_rows.dropna(subset=["yhat"]).copy()
        fut_rows["yhat"] = fut_rows["yhat"].clip(lower=0.0)   # CPU% ≥ 0
        fut_rows = fut_rows.head(total_hours).copy()

        if fut_rows.empty:
            logger.warning("No valid future forecast rows extracted.")
            return {"minutely": pd.DataFrame(), "hourly": pd.DataFrame(), "daily": pd.DataFrame()}

        logger.info(
            "Capacity model forecast: %d hourly rows (from %s to %s).",
            len(fut_rows), fut_rows["ds"].min(), fut_rows["ds"].max(),
        )

        # 5. Build hourly output (mean = point forecast; ±1.5σ of hourly historical σ for range)
        hourly_std = float(df_hourly["y"].std())
        fut_hourly = fut_rows[["ds", "yhat"]].copy()
        fut_hourly["yhat_mean"] = fut_hourly["yhat"]
        fut_hourly["yhat_max"]  = (fut_hourly["yhat"] + hourly_std).clip(lower=0.0)
        fut_hourly["yhat_min"]  = (fut_hourly["yhat"] - hourly_std).clip(lower=0.0)
        fut_hourly = fut_hourly[["ds", "yhat_mean", "yhat_max", "yhat_min"]]

        # 6. Aggregate to daily
        fut_daily = (
            fut_hourly.set_index("ds")["yhat_mean"]
            .resample("D")
            .agg(yhat_mean="mean", yhat_max="max", yhat_min="min")
            .reset_index()
        )
        # Daily max/min from ±σ bounds
        fut_daily_bounds = (
            fut_hourly.set_index("ds")
            .resample("D")
            .agg({"yhat_max": "max", "yhat_min": "min"})
            .reset_index()
        )
        fut_daily["yhat_max"] = fut_daily_bounds["yhat_max"].values
        fut_daily["yhat_min"] = fut_daily_bounds["yhat_min"].values

        logger.info(
            "Capacity forecast ready: %d hourly rows, %d daily rows.",
            len(fut_hourly), len(fut_daily),
        )
        return {"minutely": pd.DataFrame(), "hourly": fut_hourly, "daily": fut_daily}



    def plot_capacity_forecast(
        self,
        df: pd.DataFrame,
        capacity_forecasts: dict[str, pd.DataFrame],
        entity_id: str,
        output_dir: Path | None = None,
        forecast_days: int = 30,
    ) -> dict[str, Path]:
        """
        Produce two publication-quality forecast charts:
          1. Hourly aggregated forecast for the next N days
          2. Daily aggregated forecast for the next N days

        Returns dict mapping 'hourly' -> Path, 'daily' -> Path.
        """
        import matplotlib.dates as mdates

        safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
        out_dir = output_dir or ES_FORECASTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, Path] = {}

        fut_hourly = capacity_forecasts.get("hourly", pd.DataFrame())
        fut_daily = capacity_forecasts.get("daily", pd.DataFrame())

        # ---------- Hourly chart ----------
        if not fut_hourly.empty:
            hourly_path = out_dir / f"capacity_hourly_{safe_entity}.png"
            fig, ax = plt.subplots(figsize=(18, 6), dpi=150)

            # Last 48 hours of history for context
            hist_tail = df.set_index("ds")["y"].resample("1h").mean().reset_index().tail(48)
            ax.plot(hist_tail["ds"], hist_tail["y"],
                    label="Historical (hourly mean)", color="#1f77b4", linewidth=1.8)

            # Forecast band (min-max shading)
            ax.fill_between(
                fut_hourly["ds"],
                fut_hourly["yhat_min"],
                fut_hourly["yhat_max"],
                alpha=0.18,
                color="#ff7f0e",
                label="Forecast range (min–max per hour)",
            )
            ax.plot(fut_hourly["ds"], fut_hourly["yhat_mean"],
                    label="Forecast (hourly mean)", color="#ff7f0e",
                    linewidth=2.0, linestyle="--")

            # Forecast boundary
            ax.axvline(x=df["ds"].max(), color="#d62728", linestyle=":",
                       linewidth=1.8, label="Forecast start")

            # Annotations
            mean_f = fut_hourly["yhat_mean"].mean()
            max_f = fut_hourly["yhat_max"].max()
            min_f = fut_hourly["yhat_min"].min()
            text = (
                f"{forecast_days}-Day Hourly Forecast:\n"
                f"Mean: {mean_f:.1f}%\nPeak: {max_f:.1f}%\nFloor: {min_f:.1f}%"
            )
            ax.text(0.01, 0.97, text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e7",
                              edgecolor="#ffcc80", alpha=0.95))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

            ax.set_title(
                f"{forecast_days}-Day CPU Forecast (Hourly Aggregation, 1-min model) — {entity_id}",
                fontsize=13, fontweight="bold", pad=12,
            )
            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel("CPU Usage (%)", fontsize=11)
            all_y = pd.concat([hist_tail["y"], fut_hourly["yhat_max"]], ignore_index=True)
            ax.set_ylim(bottom=0.0, top=min(120.0, all_y.max() + 10.0))
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=9)

            fig.tight_layout()
            fig.savefig(hourly_path)
            plt.close(fig)
            output_paths["hourly"] = hourly_path
            logger.info("Saved hourly capacity forecast chart: %s", hourly_path)

        # ---------- Daily chart ----------
        if not fut_daily.empty:
            daily_path = out_dir / f"capacity_daily_{safe_entity}.png"
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

            # Last 7 days of daily history for context
            hist_daily = df.set_index("ds")["y"].resample("D").mean().reset_index().tail(7)
            ax.bar(hist_daily["ds"], hist_daily["y"],
                   width=0.6, label="Historical (daily mean)",
                   color="#1f77b4", alpha=0.75)

            # Future: error bars for min/max, bars for mean
            bar_width = 0.6
            ax.bar(fut_daily["ds"], fut_daily["yhat_mean"],
                   width=bar_width, label="Forecast (daily mean)",
                   color="#ff7f0e", alpha=0.8)
            ax.errorbar(
                fut_daily["ds"],
                fut_daily["yhat_mean"],
                yerr=[
                    fut_daily["yhat_mean"] - fut_daily["yhat_min"],
                    fut_daily["yhat_max"] - fut_daily["yhat_mean"],
                ],
                fmt="none",
                color="#7f7f7f",
                capsize=3,
                linewidth=1.2,
                label="Daily min–max range",
            )

            ax.axvline(x=df["ds"].max(), color="#d62728", linestyle=":",
                       linewidth=1.8, label="Forecast start")

            mean_f = fut_daily["yhat_mean"].mean()
            max_f = fut_daily["yhat_max"].max()
            text = (
                f"{forecast_days}-Day Daily Summary:\n"
                f"Avg daily mean: {mean_f:.1f}%\n"
                f"Peak (any hour): {max_f:.1f}%"
            )
            ax.text(0.01, 0.97, text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff",
                              edgecolor="#90caf9", alpha=0.95))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

            ax.set_title(
                f"{forecast_days}-Day CPU Forecast (Daily Aggregation, 1-min model) — {entity_id}",
                fontsize=13, fontweight="bold", pad=12,
            )
            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel("CPU Usage (%) — Mean per Day", fontsize=11)
            all_y = pd.concat([hist_daily["y"], fut_daily["yhat_max"]], ignore_index=True)
            ax.set_ylim(bottom=0.0, top=min(120.0, all_y.max() + 15.0))
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=9)

            fig.tight_layout()
            fig.savefig(daily_path)
            plt.close(fig)
            output_paths["daily"] = daily_path
            logger.info("Saved daily capacity forecast chart: %s", daily_path)

        return output_paths

    def plot_forecast(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame,
        entity_id: str,
        output_file: Path | None = None,
    ) -> Path:
        """
        Plot historical metrics along with multi-step future forecasts.
        """
        safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
        output_file = output_file or (
            ES_FORECASTS_DIR / f"forecast_{safe_entity}.png"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Extract clean future predictions DataFrame
        fut_df = extract_es_forecast_rows(forecast, n_forecasts=self.n_forecasts)

        fig, ax = plt.subplots(figsize=(15, 6), dpi=150)

        # Plot historical data (last 500 minutes for high visual resolution)
        plot_df = df.tail(500)
        ax.plot(plot_df["ds"], plot_df["y"], label="Historical Actual CPU %", color="#1f77b4", linewidth=1.5)

        if not fut_df.empty:
            # Connect last historical data point to first future point for continuity
            last_hist_row = plot_df.iloc[-1]
            connected_fut_ds = pd.concat([pd.Series([last_hist_row["ds"]]), fut_df["ds"]], ignore_index=True)
            connected_fut_yhat = pd.concat([pd.Series([last_hist_row["y"]]), fut_df["yhat"]], ignore_index=True)

            # Plot future forecast line
            ax.plot(
                connected_fut_ds,
                connected_fut_yhat,
                label=f"Forecasted CPU % ({self.n_forecasts} min ahead)",
                color="#ff7f0e",
                linewidth=2.2,
                linestyle="--",
                marker="o",
                markersize=3,
                markevery=max(1, len(fut_df) // 15),
            )

            # Mark forecast boundary
            boundary_time = last_hist_row["ds"]
            ax.axvline(
                x=boundary_time,
                color="#d62728",
                linestyle=":",
                linewidth=1.8,
                label="Forecast Horizon Start",
            )

            # Add forecast statistics annotation box
            min_f = fut_df["yhat"].min()
            max_f = fut_df["yhat"].max()
            mean_f = fut_df["yhat"].mean()
            text_str = f"Forecast ({self.n_forecasts}m ahead):\nMin:  {min_f:.1f}%\nMax: {max_f:.1f}%\nMean: {mean_f:.1f}%"
            ax.text(
                0.98,
                0.95,
                text_str,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
            )

        ax.set_title(
            f"1-Minute NeuralProphet CPU Forecast - Entity: {entity_id}",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Timestamp (ds)", fontsize=11)
        ax.set_ylabel("CPU Usage (%)", fontsize=11)

        # Smart y-axis limits
        max_y = max(df["y"].max(), fut_df["yhat"].max() if not fut_df.empty else 100.0)
        ax.set_ylim(bottom=0.0, top=min(110.0, max_y + 10.0))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)

        logger.info("Saved forecast plot image to: %s", output_file)
        return output_file
