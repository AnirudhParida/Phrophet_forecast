"""
pipeline/ingestion.py
=====================
Backward-compatibility forwarder to ``pipeline.ingestion.csv_ingestion``.
"""

from pipeline.ingestion.csv_ingestion import (
    describe_dataset,
    load_and_preprocess,
    parse_metric_value,
    _build_host_dataframe,
    _read_excel,
    _validate,
)

__all__ = [
    "load_and_preprocess",
    "describe_dataset",
    "parse_metric_value",
    "_build_host_dataframe",
    "_read_excel",
    "_validate",
]
