"""
pipeline/es_forecaster.py
=========================
Backward-compatibility forwarder to ``pipeline.models.neuralprophet.np_es``.
"""

from pipeline.models.neuralprophet.np_es import (
    ESMetricsForecaster,
    extract_es_forecast_rows,
)

__all__ = [
    "ESMetricsForecaster",
    "extract_es_forecast_rows",
]
