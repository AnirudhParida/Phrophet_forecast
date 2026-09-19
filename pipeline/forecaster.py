"""
pipeline/forecaster.py
======================
Backward-compatibility forwarder to ``pipeline.models.neuralprophet.np_csv``.
"""

from pipeline.models.neuralprophet.np_csv import (
    ServerMetricsForecaster,
    _REGRESSOR_MAP,
)

__all__ = [
    "ServerMetricsForecaster",
    "_REGRESSOR_MAP",
]
