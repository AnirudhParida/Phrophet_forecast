"""
pipeline/__init__.py — public API of the pipeline package.

Imports are kept lazy so that ``pipeline.ingestion`` and
``pipeline.evaluation`` can be imported in test environments where
NeuralProphet / torch are not yet installed.
"""

from pipeline.ingestion import load_and_preprocess, describe_dataset

__all__ = [
    "load_and_preprocess",
    "describe_dataset",
]


def __getattr__(name: str):
    """Lazy import of heavy forecaster / visualization utilities."""
    if name == "ServerMetricsForecaster":
        from pipeline.forecaster import ServerMetricsForecaster  # noqa: PLC0415
        return ServerMetricsForecaster
    if name == "extract_forecast_rows":
        from pipeline.visualization import extract_forecast_rows  # noqa: PLC0415
        return extract_forecast_rows
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")

