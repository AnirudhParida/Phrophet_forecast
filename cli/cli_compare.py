"""
cli/cli_compare.py
==================
Command-line interface for side-by-side model benchmarking (NeuralProphet vs Holt-Winters)
on identical temporal holdout splits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config.settings import (
    ES_CPU_CORES,
    ES_ENTITY_ID,
    ES_HOST,
    ES_INDEX,
    ES_LOOKBACK_DAYS,
    ES_METRIC_TABLE,
    ES_PASSWORD,
    ES_USER,
    HOLDOUT_DAYS,
    HOSTS,
)
from pipeline.ingestion import load_and_preprocess, load_from_elasticsearch
from pipeline.common.comparison import compare_models_csv, compare_models_es

logger = logging.getLogger("cli_compare")


def cmd_compare_csv(args: argparse.Namespace) -> None:
    """Benchmark NeuralProphet vs Holt-Winters on CSV host metrics."""
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    holdout = getattr(args, "holdout", HOLDOUT_DAYS)

    for host in target_hosts:
        if host not in datasets:
            continue
        compare_models_csv(datasets[host], host_alias=host, holdout_days=holdout)


def cmd_compare_es(args: argparse.Namespace) -> None:
    """Benchmark NeuralProphet vs Holt-Winters on ES metric data."""
    csv_path = getattr(args, "csv_path", None)
    resolution = getattr(args, "resolution", "1min")

    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided CSV path does not exist: {path}")
        df = pd.read_csv(path)
        df["ds"] = pd.to_datetime(df["ds"])
        if resolution != "1min":
            freq = "1h" if resolution == "1h" else "D"
            df = df.set_index("ds").resample(freq).mean().dropna().reset_index()
    else:
        df = load_from_elasticsearch(
            es_host=getattr(args, "host", ES_HOST),
            es_user=getattr(args, "user", ES_USER),
            es_password=getattr(args, "password", ES_PASSWORD),
            es_index=getattr(args, "index", ES_INDEX),
            entity_id=getattr(args, "entity", ES_ENTITY_ID),
            metric_table=getattr(args, "metric_table", ES_METRIC_TABLE),
            lookback_days=getattr(args, "days", ES_LOOKBACK_DAYS),
            resolution=resolution,
            cpu_cores=getattr(args, "cores", ES_CPU_CORES),
        )

    holdout = getattr(args, "holdout", None)
    compare_models_es(df, entity_id=getattr(args, "entity", ES_ENTITY_ID), resolution=resolution, holdout=holdout)


def build_compare_parser(subparsers=None) -> argparse.ArgumentParser:
    """Build Model Comparison CLI parser."""
    if subparsers is not None:
        p = subparsers.add_parser("compare", help="Side-by-side benchmark comparison (NeuralProphet vs Holt-Winters)")
    else:
        p = argparse.ArgumentParser(prog="main.py compare", description="Model Comparison CLI")

    pipe_subs = p.add_subparsers(dest="pipeline", required=True)

    # --- CSV Comparison ---
    csv_p = pipe_subs.add_parser("csv", help="Benchmark models on CSV host metrics")
    csv_p.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    csv_p.add_argument("--all", action="store_true", help="Compare all hosts")
    csv_p.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout days")
    csv_p.set_defaults(func=cmd_compare_csv)

    # --- ES Comparison ---
    es_p = pipe_subs.add_parser("es", help="Benchmark models on Elasticsearch metrics")
    es_p.add_argument("--resolution", default="1min", choices=["1min", "1h", "D"], help="Data resolution")
    es_p.add_argument("--csv-path", default=None, help="Optional CSV file path")
    es_p.add_argument("--days", type=int, default=ES_LOOKBACK_DAYS, help="Lookback days for live ES")
    es_p.add_argument("--entity", default=ES_ENTITY_ID, help="Entity ID")
    es_p.add_argument("--host", default=ES_HOST, help="ES Host URL")
    es_p.add_argument("--user", default=ES_USER, help="ES User")
    es_p.add_argument("--password", default=ES_PASSWORD, help="ES Password")
    es_p.add_argument("--index", default=ES_INDEX, help="ES Index")
    es_p.add_argument("--metric-table", default=ES_METRIC_TABLE, help="Metric table")
    es_p.add_argument("--cores", type=float, default=ES_CPU_CORES, help="Host CPU core count")
    es_p.add_argument("--holdout", type=int, default=None, help="Holdout steps")
    es_p.set_defaults(func=cmd_compare_es)

    return p
