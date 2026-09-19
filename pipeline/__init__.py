"""
pipeline/__init__.py
====================
Top-level public API for the infrastructure forecasting framework.

Architecture:
- Ingestion: ``pipeline.ingestion`` (csv_ingestion, es_ingestion)
- Models   : ``pipeline.models`` (neuralprophet, holt_winters)
- Common   : ``pipeline.common`` (evaluation, comparison, visualization)
"""

from pipeline.ingestion import (
    describe_dataset,
    describe_es_dataset,
    export_es_dataset,
    load_and_preprocess,
    load_csv_metrics,
    load_es_metrics,
    load_from_elasticsearch,
)

__all__ = [
    "load_and_preprocess",
    "load_csv_metrics",
    "describe_dataset",
    "load_from_elasticsearch",
    "load_es_metrics",
    "describe_es_dataset",
    "export_es_dataset",
    "ServerMetricsForecaster",
    "ESMetricsForecaster",
    "HoltWintersForecaster",
    "ESHoltWintersForecaster",
    "compare_models_csv",
    "compare_models_es",
    "extract_forecast_rows",
    "plot_forecast",
    "plot_overview",
    "compute_metrics",
    "format_evaluation_table",
]


def __getattr__(name: str):
    """Lazy imports for heavy forecaster and visualization components."""
    if name == "ServerMetricsForecaster":
        from pipeline.models.neuralprophet import ServerMetricsForecaster
        return ServerMetricsForecaster
    if name == "ESMetricsForecaster":
        from pipeline.models.neuralprophet import ESMetricsForecaster
        return ESMetricsForecaster
    if name == "HoltWintersForecaster":
        from pipeline.models.holt_winters import HoltWintersForecaster
        return HoltWintersForecaster
    if name == "ESHoltWintersForecaster":
        from pipeline.models.holt_winters import ESHoltWintersForecaster
        return ESHoltWintersForecaster
    if name == "compare_models_csv":
        from pipeline.common.comparison import compare_models_csv
        return compare_models_csv
    if name == "compare_models_es":
        from pipeline.common.comparison import compare_models_es
        return compare_models_es
    if name == "extract_forecast_rows":
        from pipeline.common.visualization import extract_forecast_rows
        return extract_forecast_rows
    if name == "plot_forecast":
        from pipeline.common.visualization import plot_forecast
        return plot_forecast
    if name == "plot_overview":
        from pipeline.common.visualization import plot_overview
        return plot_overview
    if name == "compute_metrics":
        from pipeline.common.evaluation import compute_metrics
        return compute_metrics
    if name == "format_evaluation_table":
        from pipeline.common.evaluation import format_evaluation_table
        return format_evaluation_table
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
