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
    """Lazy import of ServerMetricsForecaster (requires torch/NeuralProphet)."""
    if name == "ServerMetricsForecaster":
        from pipeline.forecaster import ServerMetricsForecaster  # noqa: PLC0415
        return ServerMetricsForecaster
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
