"""
pipeline/evaluation.py
======================
Backward-compatibility forwarder to ``pipeline.common.evaluation``.
"""

from pipeline.common.evaluation import (
    compute_metrics,
    format_evaluation_table,
    mae,
    mape,
    rmse,
    train_holdout_split,
    wape,
)

__all__ = [
    "mae",
    "rmse",
    "wape",
    "mape",
    "compute_metrics",
    "format_evaluation_table",
    "train_holdout_split",
]
