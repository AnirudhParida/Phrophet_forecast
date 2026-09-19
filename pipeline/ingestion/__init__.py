"""
pipeline/ingestion/__init__.py
==============================
Ingestion pipelines for time-series infrastructure forecasting:
1. CSV / Excel Dynatrace host metrics (CPU %, Memory %, Disk %, Disk Read/Write bytes)
2. Elasticsearch metric streams (1min, hourly, daily resolution)
"""

from pipeline.ingestion.csv_ingestion import (
    _build_host_dataframe,
    _read_excel,
    _validate,
    describe_dataset,
    load_and_preprocess,
    parse_metric_value,
)
from pipeline.ingestion.es_ingestion import (
    describe_es_dataset,
    export_es_dataset,
    load_from_elasticsearch,
)

# Friendly aliases
load_csv_metrics = load_and_preprocess
load_es_metrics = load_from_elasticsearch

__all__ = [
    "load_and_preprocess",
    "load_csv_metrics",
    "describe_dataset",
    "parse_metric_value",
    "_build_host_dataframe",
    "_read_excel",
    "_validate",
    "load_from_elasticsearch",
    "load_es_metrics",
    "describe_es_dataset",
    "export_es_dataset",
]
