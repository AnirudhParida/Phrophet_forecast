"""
pipeline/common/comparison.py
=============================
Cross-model benchmarking engine for evaluating forecasting models side-by-side
on identical temporal holdout splits.

Results are printed to stdout and saved to ``outputs/evaluations/{csv,es}/``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import (
    EVALUATIONS_CSV_DIR,
    EVALUATIONS_ES_DIR,
    HOLDOUT_DAYS,
    HOLDOUT_DAYS_ES,
    HOLDOUT_HOURS_ES,
    HOLDOUT_MINUTES_ES,
    METRICS,
)

logger = logging.getLogger(__name__)


def compare_models_csv(
    df: pd.DataFrame,
    host_alias: str,
    holdout_days: int = HOLDOUT_DAYS,
    metrics: Optional[list[str]] = None,
    output_dir: Path = EVALUATIONS_CSV_DIR,
) -> pd.DataFrame:
    """
    Train and evaluate NeuralProphet and Holt-Winters on the identical
    trailing holdout window, and save the benchmark summary.
    """
    from pipeline.models.neuralprophet import ServerMetricsForecaster
    from pipeline.models.holt_winters import HoltWintersForecaster

    target_metrics = metrics or [m for m in METRICS if m in df.columns]
    logger.info("Benchmarking models for %s on %d-day holdout...", host_alias, holdout_days)

    # 1. NeuralProphet evaluation
    np_forecaster = ServerMetricsForecaster()
    np_forecaster.load_and_preprocess()
    logger.info("Fitting NeuralProphet on holdout split...")
    np_results = np_forecaster.evaluate(host_name=host_alias, holdout_days=holdout_days)

    # 2. Holt-Winters evaluation
    hw_forecaster = HoltWintersForecaster()
    logger.info("Fitting Holt-Winters on holdout split...")
    hw_results = hw_forecaster.evaluate(df, holdout_days=holdout_days, metrics=target_metrics)

    # 3. Build comparison summary table
    comparison_rows = []
    lines = []
    lines.append("=" * 90)
    lines.append(f"  MODEL COMPARISON: NeuralProphet vs Holt-Winters ({host_alias} | Holdout: {holdout_days} days)")
    lines.append("=" * 90)
    lines.append(f"  {'Metric':<18} {'NP MAE':>10} {'HW MAE':>10} {'NP RMSE':>10} {'HW RMSE':>10} {'NP WAPE':>9} {'HW WAPE':>9} {'Winner':>12}")
    lines.append("  " + "-" * 86)

    for m in target_metrics:
        np_mae = np_results[m]["MAE"]
        hw_mae = hw_results[m]["MAE"]
        np_rmse = np_results[m]["RMSE"]
        hw_rmse = hw_results[m]["RMSE"]
        np_wape = np_results[m].get("WAPE", 0.0)
        hw_wape = hw_results[m].get("WAPE", 0.0)

        # Winner determined by lower RMSE
        winner = "Holt-Winters" if hw_rmse < np_rmse else "NeuralProphet"

        if m.endswith("_bytes"):
            np_m_str = f"{np_mae / 1024:.1f}KB"
            hw_m_str = f"{hw_mae / 1024:.1f}KB"
            np_r_str = f"{np_rmse / 1024:.1f}KB"
            hw_r_str = f"{hw_rmse / 1024:.1f}KB"
        else:
            np_m_str = f"{np_mae:.2f}pp"
            hw_m_str = f"{hw_mae:.2f}pp"
            np_r_str = f"{np_rmse:.2f}pp"
            hw_r_str = f"{hw_rmse:.2f}pp"

        row_str = f"  {m:<18} {np_m_str:>10} {hw_m_str:>10} {np_r_str:>10} {hw_r_str:>10} {np_wape:>8.2f}% {hw_wape:>8.2f}% {winner:>12}"
        lines.append(row_str)

        comparison_rows.append({
            "metric": m,
            "np_mae": np_mae,
            "hw_mae": hw_mae,
            "np_rmse": np_rmse,
            "hw_rmse": hw_rmse,
            "np_wape": np_wape,
            "hw_wape": hw_wape,
            "winner": winner,
        })

    lines.append("=" * 90)
    table_text = "\n".join(lines)
    print("\n" + table_text + "\n")

    res_df = pd.DataFrame(comparison_rows)

    # Export report to outputs/evaluations/csv/
    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv = output_dir / f"benchmark_{host_alias}.csv"
    report_txt = output_dir / f"benchmark_{host_alias}.txt"
    res_df.to_csv(report_csv, index=False)
    report_txt.write_text(table_text)
    logger.info("Saved benchmark reports to: %s and %s", report_csv, report_txt)

    return res_df


def compare_models_es(
    df: pd.DataFrame,
    entity_id: str,
    resolution: str = "1min",
    holdout: Optional[int] = None,
    output_dir: Path = EVALUATIONS_ES_DIR,
) -> pd.DataFrame:
    """
    Train and evaluate NeuralProphet and Holt-Winters on the ES dataset
    and save the benchmark report.
    """
    from pipeline.models.neuralprophet import ESMetricsForecaster
    from pipeline.models.holt_winters import ESHoltWintersForecaster

    logger.info("Benchmarking ES models for entity %s (resolution=%s)...", entity_id, resolution)

    actual_holdout = holdout or (
        HOLDOUT_MINUTES_ES if resolution == "1min"
        else (HOLDOUT_HOURS_ES if resolution == "1h" else HOLDOUT_DAYS_ES)
    )

    # 1. NeuralProphet evaluation
    np_fc = ESMetricsForecaster(freq="1min" if resolution == "1min" else ("1h" if resolution == "1h" else "D"))
    logger.info("Evaluating NeuralProphet on %d steps...", actual_holdout)
    np_res = np_fc.evaluate(df, holdout_minutes=actual_holdout)

    # 2. Holt-Winters evaluation
    hw_fc = ESHoltWintersForecaster(resolution=resolution)
    logger.info("Evaluating Holt-Winters on %d steps...", actual_holdout)
    hw_res = hw_fc.evaluate(df, holdout_steps=actual_holdout)

    np_mae = np_res["mae"]
    hw_mae = hw_res["MAE"]
    np_rmse = np_res["rmse"]
    hw_rmse = hw_res["RMSE"]
    hw_wape = hw_res.get("WAPE", 0.0)

    winner = "Holt-Winters" if hw_rmse < np_rmse else "NeuralProphet"

    lines = []
    lines.append("=" * 80)
    lines.append(f"  ES BENCHMARK: NeuralProphet vs Holt-Winters ({resolution} | Holdout: {actual_holdout})")
    lines.append("=" * 80)
    lines.append(f"  {'Model':<18} {'MAE':>14} {'RMSE':>14} {'WAPE':>12}")
    lines.append("  " + "-" * 76)
    lines.append(f"  {'NeuralProphet':<18} {np_mae:>12.4f} pp {np_rmse:>12.4f} pp {'N/A':>12}")
    lines.append(f"  {'Holt-Winters':<18} {hw_mae:>12.4f} pp {hw_rmse:>12.4f} pp {hw_wape:>11.2f}%")
    lines.append("  " + "-" * 76)
    lines.append(f"  Winner (by RMSE): {winner}")
    lines.append("=" * 80)

    table_text = "\n".join(lines)
    print("\n" + table_text + "\n")

    res_df = pd.DataFrame([{
        "entity_id": entity_id,
        "resolution": resolution,
        "holdout": actual_holdout,
        "np_mae": np_mae,
        "hw_mae": hw_mae,
        "np_rmse": np_rmse,
        "hw_rmse": hw_rmse,
        "hw_wape": hw_wape,
        "winner": winner,
    }])

    # Export report to outputs/evaluations/es/
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
    report_csv = output_dir / f"benchmark_es_{safe_entity}_{resolution}.csv"
    report_txt = output_dir / f"benchmark_es_{safe_entity}_{resolution}.txt"
    res_df.to_csv(report_csv, index=False)
    report_txt.write_text(table_text)
    logger.info("Saved ES benchmark reports to: %s and %s", report_csv, report_txt)

    return res_df
