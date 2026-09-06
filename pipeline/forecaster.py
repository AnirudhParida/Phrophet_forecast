"""
pipeline/forecaster.py
=======================
Core ``ServerMetricsForecaster`` class — the primary interface for the
NeuralProphet infrastructure forecasting pipeline.

Architecture
------------
Six independent NeuralProphet model instances are trained (2 hosts × 3
metrics).  Each model uses the *other two* metrics of the same host as lagged
cross-regressors, capturing inter-metric correlations within a single server
(e.g., higher memory pressure often coincides with elevated CPU).

Model per metric
----------------
  CPU model    : target=cpu_pct,    regressors=[memory_pct, disk_pct]
  Memory model : target=memory_pct, regressors=[cpu_pct, disk_pct]
  Disk model   : target=disk_pct,   regressors=[cpu_pct, memory_pct]

Lagged regressor strategy for multi-step forecasting
------------------------------------------------------
NeuralProphet's ``add_lagged_regressor()`` uses *past* regressor values (lags),
not future ones.  When generating a 7-step-ahead forecast:

  - The AR component predicts future y values auto-regressively.
  - The lagged regressors use the lookback window of n_lags days from the last
    known observation.
  - For future forecast rows where regressor values are unknown, we fill them
    with the rolling mean of the last ``n_lags`` actual values.  This is a
    pragmatic naive regressor forecast that avoids the circular-dependency
    problem of forecasting regressors themselves.

This approach is documented inline so that users can substitute a more
sophisticated regressor-forecasting strategy if needed.

Model persistence
-----------------
After fitting, each model is saved to ``models/saved/{host}_{metric}_lags{n}.np``
using NeuralProphet's native ``save()`` API.  ``predict()`` automatically loads
the saved model if one exists, so retraining is not required for inference.

Parallel fitting
----------------
``fit(host_name=None, parallel=True)`` uses ``concurrent.futures.ProcessPoolExecutor``
to fit both hosts concurrently.  Within a single host, the three metric models
are fitted sequentially (PyTorch's own threading is already used internally).
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    AR_LAYERS,
    BATCH_SIZE,
    CAP_VALUE,
    COUNTRY_HOLIDAYS,
    DAILY_SEASONALITY,
    EPOCHS,
    FLOOR_VALUE,
    FORECASTS_DIR,
    FREQ,
    GROWTH,
    HOLDOUT_DAYS,
    HOSTS,
    LEARNING_RATE,
    LOSS_FUNC,
    MAX_WORKERS,
    METRIC_N_LAGS,
    METRICS,
    MODELS_DIR,
    N_FORECASTS,
    N_LAGS_DEFAULT,
    NORMALIZE,
    QUANTILES,
    RANDOM_SEED,
    TWO_STAGE_REGRESSORS,
    WEEKLY_SEASONALITY,
    YEARLY_SEASONALITY,
)
from pipeline.evaluation import (
    compare_lag_results,
    compute_metrics,
    format_evaluation_table,
    train_holdout_split,
)
from pipeline.ingestion import load_and_preprocess
from pipeline.visualization import plot_forecast, plot_overview

logger = logging.getLogger(__name__)

# Map each target metric to its cross-regressor columns
_REGRESSOR_MAP: dict[str, list[str]] = {
    "cpu_pct":          ["memory_pct", "disk_pct", "disk_read_bytes", "disk_write_bytes"],
    "memory_pct":       ["cpu_pct", "disk_pct", "disk_read_bytes", "disk_write_bytes"],
    "disk_pct":         ["cpu_pct", "memory_pct", "disk_read_bytes", "disk_write_bytes"],
    "disk_read_bytes":  ["cpu_pct", "memory_pct", "disk_pct", "disk_write_bytes"],
    "disk_write_bytes": ["cpu_pct", "memory_pct", "disk_pct", "disk_read_bytes"],
}


# ---------------------------------------------------------------------------
# Helper: seed all random sources for reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # deferred: only required when NeuralProphet is installed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not installed yet; seeds only Python/NumPy


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ServerMetricsForecaster:
    """
    Production-grade forecasting pipeline for server infrastructure metrics.

    Usage
    -----
    >>> forecaster = ServerMetricsForecaster()
    >>> forecaster.load_and_preprocess()           # ingest Excel files
    >>> forecaster.fit("HYDUPINTAPP16", n_lags=7)  # train 3 models
    >>> forecaster.predict("HYDUPINTAPP16")         # 7-day forecast charts
    >>> forecaster.evaluate("HYDUPINTAPP16")        # holdout MAE/RMSE/MAPE

    Parameters
    ----------
    n_lags      : AR lookback window (7 = 1 week, 14 = 2 weeks).
    n_forecasts : Number of steps ahead to forecast (default 7 days).
    """

    def __init__(
        self,
        n_lags: int = N_LAGS_DEFAULT,
        n_forecasts: int = N_FORECASTS,
    ) -> None:
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self._host_data: dict[str, pd.DataFrame] = {}   # populated by load_and_preprocess
        self._models: dict[str, dict[str, object]] = {}  # {host: {metric: NeuralProphet}}
        _set_seed()

    # ------------------------------------------------------------------
    # 1. Data ingestion
    # ------------------------------------------------------------------

    def load_and_preprocess(
        self,
        cpu_file: Optional[Path] = None,
        mem_file: Optional[Path] = None,
        disk_file: Optional[Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Ingest the three Excel exports and cache per-host DataFrames.

        Returns
        -------
        Dict[host_alias → DataFrame[ds, cpu_pct, memory_pct, disk_pct]]
        """
        kwargs = {}
        if cpu_file:
            kwargs["cpu_file"] = cpu_file
        if mem_file:
            kwargs["mem_file"] = mem_file
        if disk_file:
            kwargs["disk_file"] = disk_file

        self._host_data = load_and_preprocess(**kwargs)
        logger.info("Loaded data for hosts: %s", list(self._host_data.keys()))
        return self._host_data

    def _resolve_n_lags(self, metric: str, override_n_lags: Optional[int] = None) -> int:
        """Resolve metric-specific n_lags lookback window."""
        if override_n_lags is not None:
            return override_n_lags
        return METRIC_N_LAGS.get(metric, self.n_lags)

    # ------------------------------------------------------------------
    # 2. Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        host_name: Optional[str] = None,
        n_lags: Optional[int] = None,
        parallel: bool = False,
    ) -> None:
        """
        Train NeuralProphet models for one or all hosts.

        Parameters
        ----------
        host_name : Short host alias (e.g. ``"HYDUPINTAPP16"``).
                    If None, trains all hosts.
        n_lags    : Override instance-level n_lags for this run.
        parallel  : If True and host_name is None, fit hosts in parallel
                    using ``ProcessPoolExecutor``.
        """
        if not self._host_data:
            raise RuntimeError(
                "No data loaded.  Call load_and_preprocess() first."
            )

        if host_name is not None:
            self._fit_host(host_name, n_lags)
        elif parallel:
            self._fit_all_parallel(n_lags)
        else:
            for alias in self._host_data:
                self._fit_host(alias, n_lags)

    def _fit_host(self, host_alias: str, n_lags: Optional[int] = None) -> None:
        """Train all three metric models for a single host."""
        if host_alias not in self._host_data:
            raise KeyError(
                f"Host '{host_alias}' not in loaded data.  "
                f"Available: {list(self._host_data.keys())}"
            )

        df = self._host_data[host_alias]
        self._models.setdefault(host_alias, {})

        for metric in METRICS:
            metric_n_lags = self._resolve_n_lags(metric, n_lags)
            logger.info(
                "Fitting [%s | %s]  n_lags=%d  n_forecasts=%d …",
                host_alias, metric, metric_n_lags, self.n_forecasts,
            )
            model = self._build_model(metric_n_lags)
            train_df = self._prepare_np_dataframe(df, metric)
            model = self._add_regressors(model, metric, metric_n_lags)

            # NeuralProphet fit returns the model itself
            metrics_df = model.fit(
                train_df,
                freq=FREQ,
                progress="bar",   # show tqdm bar during training
            )
            logger.info(
                "[%s | %s] Final training loss: %.6f",
                host_alias,
                metric,
                metrics_df["Loss"].iloc[-1] if "Loss" in metrics_df.columns else float("nan"),
            )

            self._models[host_alias][metric] = model
            self._save_model(model, host_alias, metric, metric_n_lags)

    def _fit_all_parallel(self, n_lags: Optional[int] = None) -> None:
        """
        Fit all hosts in parallel using separate processes.
        """
        host_aliases = list(self._host_data.keys())
        logger.info(
            "Parallel fit: %d hosts on %d workers", len(host_aliases), MAX_WORKERS
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_parallel_fit_worker, alias, n_lags, self.n_forecasts): alias
                for alias in host_aliases
            }
            for future in concurrent.futures.as_completed(futures):
                alias = futures[future]
                try:
                    future.result()
                    logger.info("Parallel fit completed: %s", alias)
                except Exception as exc:
                    logger.error("Parallel fit failed for %s: %s", alias, exc)
                    raise

        # Reload saved models into instance state
        for alias in host_aliases:
            self._models.setdefault(alias, {})
            for metric in METRICS:
                metric_n_lags = self._resolve_n_lags(metric, n_lags)
                self._models[alias][metric] = self._load_model(alias, metric, metric_n_lags)

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        host_name: str,
        n_lags: Optional[int] = None,
        save_charts: bool = True,
        use_two_stage: bool = TWO_STAGE_REGRESSORS,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate n_forecasts-day ahead forecasts for all metrics of a host.

        Parameters
        ----------
        host_name     : Short host alias.
        n_lags        : Override n_lags for saved model lookup.
        save_charts   : If True, persist PNG charts to disk.
        use_two_stage : If True, use univariate Stage 1 forecasts for future regressors.

        Returns
        -------
        Dict[metric → NeuralProphet prediction DataFrame]
        """
        df = self._get_host_df(host_name)

        # Stage 1 univariate regressor predictions
        stage1_forecasts = None
        if use_two_stage:
            logger.info("Running Stage-1 univariate forecasts for regressors [%s] …", host_name)
            try:
                stage1_forecasts = self._predict_stage1_univariate(df, override_n_lags=n_lags)
            except Exception as exc:
                logger.warning("Stage-1 regressor forecast failed: %s. Falling back to rolling means.", exc)

        forecast_results: dict[str, pd.DataFrame] = {}

        for metric in METRICS:
            metric_n_lags = self._resolve_n_lags(metric, n_lags)
            self._ensure_metric_model_loaded(host_name, metric, metric_n_lags)
            model = self._models[host_name][metric]
            train_df = self._prepare_np_dataframe(df, metric)
            future_df = self._make_future_df(
                model, train_df, df, metric, metric_n_lags, stage1_forecasts=stage1_forecasts
            )

            logger.info("Predicting [%s | %s]  n_lags=%d …", host_name, metric, metric_n_lags)
            forecast = model.predict(future_df)
            forecast_results[metric] = forecast

            if save_charts:
                plot_forecast(
                    host_alias=host_name,
                    metric=metric,
                    actuals_df=df[["ds", metric]],
                    forecast_df=forecast,
                    n_lags=metric_n_lags,
                    save=True,
                )

        if save_charts:
            plot_overview(
                host_alias=host_name,
                actuals_df=df,
                forecasts=forecast_results,
                n_lags=n_lags if n_lags is not None else self.n_lags,
                save=True,
            )
            print(f"\n✅ Charts saved to: {FORECASTS_DIR}/")

        return forecast_results

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        host_name: str,
        holdout_days: int = HOLDOUT_DAYS,
        n_lags: Optional[int] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Train on all-except-holdout rows and evaluate on the holdout set.
        """
        df = self._get_host_df(host_name)
        eval_results: dict[str, dict[str, float]] = {}

        for metric in METRICS:
            metric_n_lags = self._resolve_n_lags(metric, n_lags)
            logger.info(
                "Evaluating [%s | %s] with holdout=%d days, n_lags=%d …",
                host_name, metric, holdout_days, metric_n_lags,
            )

            train_df_full, holdout_df = train_holdout_split(
                df,
                holdout_days=holdout_days,
                n_lags=metric_n_lags,
                n_forecasts=self.n_forecasts,
            )

            # Fit a temporary model on the training split
            temp_model = self._build_model(metric_n_lags)
            temp_model = self._add_regressors(temp_model, metric, metric_n_lags)
            train_np = self._prepare_np_dataframe(train_df_full, metric)
            temp_model.fit(train_np, freq=FREQ, progress="bar")

            # Predict over the holdout window using a rolling context
            actuals, predicted = self._rolling_holdout_predict(
                model=temp_model,
                train_df=train_df_full,
                holdout_df=holdout_df,
                metric=metric,
                n_lags=metric_n_lags,
            )

            eval_results[metric] = compute_metrics(
                actual=actuals,
                predicted=predicted,
                metric_name=f"{host_name}|{metric}",
            )

        print(format_evaluation_table(eval_results, host_name, n_lags if n_lags is not None else self.n_lags))
        return eval_results

    # ------------------------------------------------------------------
    # 5. Compare lags (n_lags=7 vs n_lags=14)
    # ------------------------------------------------------------------

    def compare_lags(
        self,
        host_name: str,
        holdout_days: int = HOLDOUT_DAYS,
    ) -> tuple[dict, dict]:
        """
        Train with n_lags=7 and n_lags=14, compare holdout RMSE, print winner.
        """
        print(f"\n📊  Comparing n_lags=7 vs n_lags=14 for {host_name} …\n")
        results_7 = self.evaluate(host_name, holdout_days=holdout_days, n_lags=7)
        results_14 = self.evaluate(host_name, holdout_days=holdout_days, n_lags=14)
        compare_lag_results(results_7, results_14, host_name)
        return results_7, results_14

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_stage1_univariate(
        self,
        df: pd.DataFrame,
        override_n_lags: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fit fast univariate baseline models to project future regressor trajectories."""
        stage1_forecasts = {}
        for metric in METRICS:
            metric_n_lags = self._resolve_n_lags(metric, override_n_lags)
            model = self._build_model(metric_n_lags)
            np_df = df[["ds", metric]].rename(columns={metric: "y"}).copy()
            np_df["ds"] = pd.to_datetime(np_df["ds"])
            if GROWTH == "logistic" and metric.endswith("_pct"):
                np_df["cap"] = CAP_VALUE
                np_df["floor"] = FLOOR_VALUE
            model.fit(np_df, freq=FREQ, progress="none")
            fut = model.make_future_dataframe(df=np_df, periods=self.n_forecasts, n_historic_predictions=True)
            if GROWTH == "logistic" and metric.endswith("_pct"):
                fut["cap"] = CAP_VALUE
                fut["floor"] = FLOOR_VALUE
            pred = model.predict(fut)
            stage1_forecasts[metric] = pred
        return stage1_forecasts

    @staticmethod
    def _build_model(n_lags: int) -> "NeuralProphet":
        """Construct a NeuralProphet instance with pipeline hyper-parameters."""
        from neuralprophet import NeuralProphet  # deferred import

        kwargs = dict(
            n_forecasts=N_FORECASTS,
            n_lags=n_lags,
            growth=GROWTH,
            ar_layers=AR_LAYERS,
            yearly_seasonality=YEARLY_SEASONALITY,
            weekly_seasonality=WEEKLY_SEASONALITY,
            daily_seasonality=DAILY_SEASONALITY,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            normalize=NORMALIZE,
            quantiles=QUANTILES,
        )
        # Omit loss_func when quantiles are active so NeuralProphet uses PinballLoss for quantile bounds
        if not QUANTILES and LOSS_FUNC:
            kwargs["loss_func"] = LOSS_FUNC

        model = NeuralProphet(**kwargs)
        if COUNTRY_HOLIDAYS:
            try:
                model.add_country_holidays(country_name=COUNTRY_HOLIDAYS)
            except Exception as exc:
                logger.warning("Could not add country holidays '%s': %s", COUNTRY_HOLIDAYS, exc)
        return model


    @staticmethod
    def _add_regressors(
        model: "NeuralProphet",
        target_metric: str,
        n_lags: int,
    ) -> "NeuralProphet":
        """Register cross-metric lagged regressors."""
        for reg_col in _REGRESSOR_MAP[target_metric]:
            model.add_lagged_regressor(names=reg_col, n_lags=n_lags)
        return model

    @staticmethod
    def _prepare_np_dataframe(df: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """Reformat a host DataFrame into NeuralProphet's expected schema."""
        np_df = df.rename(columns={target_metric: "y"}).copy()
        np_df["ds"] = pd.to_datetime(np_df["ds"])
        if GROWTH == "logistic" and target_metric.endswith("_pct"):
            np_df["cap"] = CAP_VALUE
            np_df["floor"] = FLOOR_VALUE
        keep_cols = ["ds", "y"] + _REGRESSOR_MAP[target_metric]
        if GROWTH == "logistic" and target_metric.endswith("_pct"):
            keep_cols.extend(["cap", "floor"])
        return np_df[keep_cols].reset_index(drop=True)

    def _make_future_df(
        self,
        model: "NeuralProphet",
        train_np_df: pd.DataFrame,
        full_host_df: pd.DataFrame,
        target_metric: str,
        n_lags: int,
        stage1_forecasts: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build the future DataFrame for NeuralProphet's predict() call."""
        future = model.make_future_dataframe(
            df=train_np_df,
            periods=self.n_forecasts,
            n_historic_predictions=True,
        )
        if GROWTH == "logistic" and target_metric.endswith("_pct"):
            future["cap"] = CAP_VALUE
            future["floor"] = FLOOR_VALUE

        regressor_cols = _REGRESSOR_MAP[target_metric]

        for reg_col in regressor_cols:
            if stage1_forecasts and reg_col in stage1_forecasts:
                s1_df = stage1_forecasts[reg_col]
                yhat_col = "yhat1" if "yhat1" in s1_df.columns else "yhat"
                if yhat_col in s1_df.columns:
                    ds_map = dict(zip(pd.to_datetime(s1_df["ds"]), s1_df[yhat_col]))
                    future[reg_col] = future["ds"].map(ds_map).fillna(future[reg_col])

            last_vals = full_host_df[reg_col].values[-n_lags:]
            rolling_mean = float(np.mean(last_vals))
            future[reg_col] = future[reg_col].fillna(rolling_mean)

        return future

    def _rolling_holdout_predict(
        self,
        model: "NeuralProphet",
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        metric: str,
        n_lags: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        actuals: list[float] = []
        predicted: list[float] = []
        holdout_days = len(holdout_df)

        context_df = train_df.copy()

        for batch_start in range(0, holdout_days, self.n_forecasts):
            batch_end = min(batch_start + self.n_forecasts, holdout_days)
            batch_size = batch_end - batch_start
            batch_holdout = holdout_df.iloc[batch_start:batch_end]

            np_context = self._prepare_np_dataframe(context_df, metric)
            future = self._make_future_df(model, np_context, context_df, metric, n_lags)

            try:
                forecast = model.predict(future)

                future_mask = forecast["y"].isna() if "y" in forecast.columns \
                    else pd.Series(
                        [False] * (len(forecast) - self.n_forecasts)
                        + [True] * self.n_forecasts,
                        index=forecast.index,
                    )
                future_rows = forecast[future_mask]

                if future_rows.empty:
                    logger.warning("No future rows in forecast for batch %d", batch_start)
                    predicted.extend([np.nan] * batch_size)
                else:
                    first_future = future_rows.iloc[0]
                    batch_preds = [
                        float(first_future.get(f"yhat{i + 1}", np.nan))
                        for i in range(batch_size)
                    ]
                    predicted.extend(batch_preds)

            except Exception as exc:
                logger.warning("Batch prediction failed at offset %d: %s", batch_start, exc)
                predicted.extend([np.nan] * batch_size)

            actuals.extend(batch_holdout[metric].tolist())
            context_df = pd.concat([context_df, batch_holdout], ignore_index=True)

        pairs = [
            (a, p)
            for a, p in zip(actuals, predicted)
            if not (np.isnan(a) or np.isnan(p))
        ]
        if not pairs:
            return np.array([]), np.array([])
        act_arr, pred_arr = zip(*pairs)
        return np.array(act_arr), np.array(pred_arr)

    @staticmethod
    def _model_path(host_alias: str, metric: str, n_lags: int) -> Path:
        return MODELS_DIR / f"{host_alias}_{metric}_lags{n_lags}.np"

    def _save_model(
        self,
        model: "NeuralProphet",
        host_alias: str,
        metric: str,
        n_lags: int,
    ) -> None:
        import torch  # deferred import

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = self._model_path(host_alias, metric, n_lags)
        torch.save(model, str(path))
        logger.info("Model saved → %s", path)

    def _load_model(
        self,
        host_alias: str,
        metric: str,
        n_lags: int,
    ) -> "NeuralProphet":
        import torch  # deferred import

        path = self._model_path(host_alias, metric, n_lags)
        if not path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {path}\n"
                f"Run 'python main.py fit --host {host_alias} --n-lags {n_lags}' first."
            )
        logger.info("Loading model ← %s", path)
        return torch.load(str(path), weights_only=False)

    def _ensure_metric_model_loaded(
        self,
        host_alias: str,
        metric: str,
        n_lags: int,
    ) -> None:
        if host_alias not in self._models:
            self._models[host_alias] = {}
        if metric not in self._models[host_alias]:
            self._models[host_alias][metric] = self._load_model(
                host_alias, metric, n_lags
            )

    def _ensure_models_loaded(self, host_alias: str, n_lags: Optional[int] = None) -> None:
        for metric in METRICS:
            metric_n_lags = self._resolve_n_lags(metric, n_lags)
            self._ensure_metric_model_loaded(host_alias, metric, metric_n_lags)

    def _get_host_df(self, host_alias: str) -> pd.DataFrame:
        if not self._host_data:
            raise RuntimeError(
                "No data loaded.  Call load_and_preprocess() first."
            )
        if host_alias not in self._host_data:
            raise KeyError(
                f"Host '{host_alias}' not found.  "
                f"Available: {list(self._host_data.keys())}"
            )
        return self._host_data[host_alias]


# ---------------------------------------------------------------------------
# Parallel worker (top-level function required for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _parallel_fit_worker(host_alias: str, n_lags: Optional[int], n_forecasts: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    logger.info("Worker started: %s", host_alias)
    forecaster = ServerMetricsForecaster(n_lags=N_LAGS_DEFAULT, n_forecasts=n_forecasts)
    forecaster.load_and_preprocess()
    forecaster.fit(host_name=host_alias, n_lags=n_lags)
    logger.info("Worker completed: %s", host_alias)
