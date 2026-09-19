"""
pipeline/es_ingestion.py
========================
Backward-compatibility forwarder to ``pipeline.ingestion.es_ingestion``.
"""

from pipeline.ingestion.es_ingestion import (
    describe_es_dataset,
    export_es_dataset,
    load_from_elasticsearch,
)

__all__ = [
    "load_from_elasticsearch",
    "describe_es_dataset",
    "export_es_dataset",
]
