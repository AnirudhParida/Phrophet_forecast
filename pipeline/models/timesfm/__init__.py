"""
pipeline/models/timesfm/__init__.py
===================================
Google TimesFM 3.0 foundation model package:
1. ``TimesFMForecaster`` (Multi-metric CSV host server pipeline)
2. ``ESTimesFMForecaster`` (Elasticsearch streaming metrics pipeline)
3. ``resolve_device`` & ``get_timesfm_model`` (Hardware device & singleton manager)
"""

from pipeline.models.timesfm.model_loader import (
    clear_timesfm_cache,
    get_timesfm_model,
    resolve_device,
)
from pipeline.models.timesfm.tfm_csv import TimesFMForecaster, _clip_metric
from pipeline.models.timesfm.tfm_es import ESTimesFMForecaster

__all__ = [
    "TimesFMForecaster",
    "ESTimesFMForecaster",
    "_clip_metric",
    "resolve_device",
    "get_timesfm_model",
    "clear_timesfm_cache",
]
