"""
pipeline/evaluation.py
=======================
Holdout evaluation utilities for the NeuralProphet forecasting pipeline.

Provides three scalar error metrics — MAE, RMSE, and MAPE — computed on a
temporal holdout set (last N days withheld from training).

Design notes
------------
- Uses a *trailing holdout* split (last ``holdout_days`` rows), not a random
  split, to honour the temporal ordering of the time series.
- MAPE is guarded against division-by-zero for near-zero actuals (relevant for
  CPU % which can be as low as 0.76 %).
- All metrics return values in percentage-point units (since ``y`` is already
  in [0, 100] percent scale), making them directly interpretable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (percentage points)."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (percentage points)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%).

    ``epsilon`` prevents division-by-zero for near-zero actuals (e.g. CPU %).
    The result is expressed as a percentage (e.g. 5.3 means 5.3 % error).
    """
    safe_actual = np.where(np.abs(actual) < epsilon, epsilon, actual)
    return float(np.mean(np.abs((actual - predicted) / safe_actual)) * 100)


# ---------------------------------------------------------------------------
# Holdout split helpers
# ---------------------------------------------------------------------------


def train_holdout_split(
    df: pd.DataFrame,
    holdout_days: int,
    n_lags: int,
    n_forecasts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a host DataFrame into training and holdout sets.

    The split respects temporal order: the last ``holdout_days`` rows form the
    holdout; everything before forms the training set.

    NeuralProphet requires at least ``n_lags + n_forecasts`` rows to create one
    valid sample.  We warn if the training set is close to this minimum.

    Parameters
    ----------
    df           : Full host DataFrame (ds, cpu_pct, memory_pct, disk_pct).
    holdout_days : Number of trailing days reserved for evaluation.
    n_lags       : AR lookback window (used for minimum-size check).
    n_forecasts  : Forecast horizon (used for minimum-size check).

    Returns
    -------
    (train_df, holdout_df) tuple of DataFrames.
    """
    if holdout_days >= len(df):
        raise ValueError(
            f"holdout_days={holdout_days} ≥ total rows={len(df)}.  "
            "Choose a smaller holdout window."
        )

    train_df = df.iloc[: -holdout_days].copy()
    holdout_df = df.iloc[-holdout_days:].copy()

    min_train = n_lags + n_forecasts
    if len(train_df) < min_train:
        raise ValueError(
            f"Training set has only {len(train_df)} rows after holdout split, "
            f"but NeuralProphet requires at least n_lags + n_forecasts = "
            f"{n_lags} + {n_forecasts} = {min_train} rows."
        )
    if len(train_df) < 2 * min_train:
        logger.warning(
            "Training set has only %d rows (≥ %d recommended). "
            "Model accuracy may be limited.",
            len(train_df),
            2 * min_train,
        )

    return train_df, holdout_df


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_name: str,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE for one metric and return as a labelled dict.

    Parameters
    ----------
    actual      : 1-D array of ground-truth values.
    predicted   : 1-D array of model predictions (same length).
    metric_name : Label used in log output (e.g. ``"cpu_pct"``).

    Returns
    -------
    dict with keys ``MAE``, ``RMSE``, ``MAPE``.
    """
    mae_val = mae(actual, predicted)
    rmse_val = rmse(actual, predicted)
    mape_val = mape(actual, predicted)

    logger.info(
        "[%s] MAE=%.4f pp | RMSE=%.4f pp | MAPE=%.2f %%",
        metric_name,
        mae_val,
        rmse_val,
        mape_val,
    )
    return {"MAE": mae_val, "RMSE": rmse_val, "MAPE": mape_val}


def format_evaluation_table(
    eval_results: dict[str, dict[str, float]],
    host_alias: str,
    n_lags: int,
) -> str:
    """
    Return a formatted ASCII table of evaluation results for a given host.

    Parameters
    ----------
    eval_results : Dict[metric_name → {MAE, RMSE, MAPE}].
    host_alias   : Short host name for the table header.
    n_lags       : n_lags setting used in training (shown in header).

    Returns
    -------
    Multi-line string suitable for printing to stdout or logging.
    """
    header = f"\n{'='*60}\n  Evaluation: {host_alias}  |  n_lags={n_lags}\n{'='*60}"
    rows = [f"  {'Metric':<15} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}"]
    rows.append(f"  {'-'*47}")
    for metric, scores in eval_results.items():
        rows.append(
            f"  {metric:<15} "
            f"{scores['MAE']:>9.4f} "
            f"{scores['RMSE']:>9.4f} "
            f"{scores['MAPE']:>9.2f}%"
        )
    rows.append(f"{'='*60}")
    return header + "\n" + "\n".join(rows)


def compare_lag_results(
    results_7: dict[str, Any],
    results_14: dict[str, Any],
    host_alias: str,
) -> None:
    """
    Print a side-by-side comparison of n_lags=7 vs n_lags=14 RMSE scores
    and print a recommendation.

    Parameters
    ----------
    results_7   : Evaluation dict from the n_lags=7 run.
    results_14  : Evaluation dict from the n_lags=14 run.
    host_alias  : Short host name for display.
    """
    print(f"\n{'='*70}")
    print(f"  Lag Comparison: {host_alias}")
    print(f"  {'Metric':<15} {'RMSE (n_lags=7)':>18} {'RMSE (n_lags=14)':>18} {'Winner':>8}")
    print(f"  {'-'*65}")

    for metric in results_7:
        r7 = results_7[metric]["RMSE"]
        r14 = results_14[metric]["RMSE"]
        winner = "n_lags=7" if r7 <= r14 else "n_lags=14"
        print(f"  {metric:<15} {r7:>18.4f} {r14:>18.4f} {winner:>8}")

    print(f"{'='*70}\n")
