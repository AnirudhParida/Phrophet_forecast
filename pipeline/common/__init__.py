"""
pipeline/common/__init__.py
===========================
Shared utilities: evaluation metrics, model comparison benchmarks, and visualization.
"""

from pipeline.common.comparison import compare_models_csv, compare_models_es
from pipeline.common.evaluation import (
    compute_metrics,
    format_evaluation_table,
    mae,
    mape,
    rmse,
    train_holdout_split,
    wape,
)
from pipeline.common.visualization import (
    extract_forecast_rows,
    plot_forecast,
    plot_overview,
)

__all__ = [
    "mae",
    "rmse",
    "wape",
    "mape",
    "compute_metrics",
    "format_evaluation_table",
    "train_holdout_split",
    "compare_models_csv",
    "compare_models_es",
    "plot_forecast",
    "plot_overview",
    "extract_forecast_rows",
]
