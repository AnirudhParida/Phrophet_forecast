"""
pipeline/es_ingestion.py
========================
Data ingestion layer for Elasticsearch infrastructure metrics.

Responsibilities
----------------
1. Connect to Elasticsearch (http://192.168.12.94:9200) with Basic Authentication.
2. Query minute-level metric data for a target entity_id and metric_table using search_after pagination.
3. Parse YYYYMMDDHHMM time_bucket integers/strings into datetime64[ns] ('ds').
4. Resample data onto a regular 1-minute grid and interpolate missing minutes.
5. Validate data quality: non-empty, values within valid bounds [0, 100], no NaNs.
6. Provide dataset inspection metrics (min, max, mean, std, row counts).
7. Export dataset to both CSV and Excel (.xlsx) formats for verification.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DATA_ES_DIR,
    DATA_EXPORT_DIR,
    ES_CPU_CORES,
    ES_ENTITY_ID,
    ES_HOST,
    ES_INDEX,
    ES_LOOKBACK_DAYS,
    ES_METRIC_TABLE,
    ES_PASSWORD,
    ES_USER,
)

logger = logging.getLogger(__name__)


def load_from_elasticsearch(
    es_host: str = ES_HOST,
    es_user: str = ES_USER,
    es_password: str = ES_PASSWORD,
    es_index: str = ES_INDEX,
    entity_id: str = ES_ENTITY_ID,
    metric_table: str = ES_METRIC_TABLE,
    lookback_days: int = ES_LOOKBACK_DAYS,
    resolution: str = "1min",
    cpu_cores: float = ES_CPU_CORES,
) -> pd.DataFrame:
    """
    Fetch minute-level metric data from Elasticsearch for a given entity_id.

    Parameters
    ----------
    es_host       : Base URL of Elasticsearch server.
    es_user       : Username for basic auth.
    es_password   : Password for basic auth.
    es_index      : Index pattern to search (e.g. netraa_metrics-all*).
    entity_id     : Entity ID (e.g. MTkyLjE2OC4xMi44Ng==.1).
    metric_table  : Metric table name (e.g. meter_vm_cpu_total_percentage).
    lookback_days : Number of past days to query.
    resolution    : Output time resolution: '1min', '1h' (hourly), or 'D' (daily).
                    Data is always fetched at 1-minute level then resampled.

    Returns
    -------
    pd.DataFrame with columns:
        - ``ds`` : datetime64[ns], at the requested resolution
        - ``y``  : float, metric value (mean across resampled interval)
    """
    logger.info("Connecting to Elasticsearch at %s ...", es_host)

    # Note: filter gte 200000000000 (12-digit minute records)
    query_body: dict[str, Any] = {
        "size": 2000,
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_id": entity_id}},
                    {"term": {"metric_table": metric_table}},
                    {"range": {"time_bucket": {"gte": 200000000000}}},
                ]
            }
        },
        "sort": [{"time_bucket": {"order": "asc"}}],
    }

    url = f"{es_host.rstrip('/')}/{es_index}/_search"
    auth_header = "Basic " + base64.b64encode(f"{es_user}:{es_password}".encode()).decode("ascii")
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header,
    }

    raw_records: list[dict[str, Any]] = []
    search_after: list[Any] | None = None

    logger.info(
        "Querying ES index '%s' for entity_id='%s', metric_table='%s'...",
        es_index,
        entity_id,
        metric_table,
    )

    while True:
        if search_after:
            query_body["search_after"] = search_after

        data_bytes = json.dumps(query_body).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            logger.error("Failed to query Elasticsearch endpoint %s: %s", url, err)
            raise ConnectionError(f"Elasticsearch query failed: {err}") from err

        hits = result.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            source = hit.get("_source", {})
            raw_records.append({
                "time_bucket": source.get("time_bucket"),
                "value": source.get("value"),
            })

        search_after = hits[-1].get("sort")
        if len(hits) < 2000:
            break

    if not raw_records:
        raise ValueError(
            f"No metric records found in Elasticsearch for entity_id='{entity_id}' "
            f"and metric_table='{metric_table}'."
        )

    logger.info("Fetched %d raw metric hits from Elasticsearch.", len(raw_records))

    df_raw = pd.DataFrame(raw_records)

    # 2. Parse time_bucket YYYYMMDDHHMM -> datetime ds
    df_raw["ds_str"] = df_raw["time_bucket"].astype(str)
    df_raw = df_raw[df_raw["ds_str"].str.len() == 12].copy()

    df_raw["ds"] = pd.to_datetime(df_raw["ds_str"], format="%Y%m%d%H%M", errors="coerce")
    raw_vals = pd.to_numeric(df_raw["value"], errors="coerce")
    if cpu_cores and cpu_cores > 1.0:
        logger.info("Normalizing CPU metric by dividing raw values by %g cores...", cpu_cores)
        df_raw["y"] = raw_vals / float(cpu_cores)
    else:
        df_raw["y"] = raw_vals

    df_raw = df_raw.dropna(subset=["ds", "y"]).copy()

    # Sort by timestamp
    df_raw = df_raw.sort_values("ds").reset_index(drop=True)

    # Filter to lookback_days if requested and data spans more than lookback_days
    if lookback_days and not df_raw.empty:
        max_ds = df_raw["ds"].max()
        cutoff_ds = max_ds - pd.Timedelta(days=lookback_days)
        df_raw = df_raw[df_raw["ds"] >= cutoff_ds].copy()

    # 3. Deduplicate identical timestamps by taking average
    df_dedup = df_raw.groupby("ds", as_index=False)["y"].mean()

    # 4. Resample onto regular 1-minute grid and interpolate missing minutes
    df_dedup = df_dedup.set_index("ds")
    df_resampled = df_dedup.resample("1min").mean()
    df_resampled["y"] = df_resampled["y"].interpolate(method="linear").ffill().bfill()

    # 5. Optionally further resample to hourly or daily resolution
    if resolution and resolution not in ("1min", "T", "min"):
        _res = resolution
        logger.info("Resampling 1-minute data to resolution '%s'...", _res)
        df_resampled = df_resampled.resample(_res).mean()
        df_resampled["y"] = df_resampled["y"].interpolate(method="linear").ffill().bfill()

    df_final = df_resampled.reset_index()

    # 6. Validation
    _validate_es_dataframe(df_final, entity_id)

    logger.info(
        "Successfully preprocessed %d rows at resolution '%s' for entity '%s' (ds: %s -> %s)",
        len(df_final),
        resolution,
        entity_id,
        df_final["ds"].min(),
        df_final["ds"].max(),
    )

    return df_final


def _validate_es_dataframe(df: pd.DataFrame, entity_id: str) -> None:
    """Fail-fast validation for ingested Elasticsearch DataFrame."""
    if df.empty:
        raise ValueError(f"DataFrame for entity '{entity_id}' is empty after preprocessing.")

    if df["y"].isna().any():
        raise ValueError(f"DataFrame for entity '{entity_id}' contains NaN values.")

    if (df["y"] < 0.0).any() or (df["y"] > 100.0).any():
        min_v = df["y"].min()
        max_v = df["y"].max()
        logger.warning(
            "Values out of standard percentage bounds [0, 100]: min=%.2f, max=%.2f", min_v, max_v
        )

    if not df["ds"].is_monotonic_increasing:
        raise ValueError(f"Timestamps for entity '{entity_id}' are not monotonically increasing.")


def describe_es_dataset(df: pd.DataFrame, entity_id: str = ES_ENTITY_ID) -> dict[str, Any]:
    """
    Print and return summary statistics for the ingested metric DataFrame.
    """
    min_val = float(df["y"].min())
    max_val = float(df["y"].max())
    mean_val = float(df["y"].mean())
    std_val = float(df["y"].std())
    total_rows = len(df)
    min_ds = df["ds"].min()
    max_ds = df["ds"].max()

    summary = {
        "entity_id": entity_id,
        "total_rows": total_rows,
        "start_time": str(min_ds),
        "end_time": str(max_ds),
        "min_value": min_val,
        "max_value": max_val,
        "mean_value": mean_val,
        "std_value": std_val,
    }

    print("\n" + "=" * 65)
    print(f" ELASTICSEARCH DATASET SUMMARY: {entity_id}")
    print("=" * 65)
    print(f" Total Rows   : {total_rows:,}")
    print(f" Time Span    : {min_ds} -> {max_ds}")
    print(f" Minimum Value: {min_val:.2f}%")
    print(f" Maximum Value: {max_val:.2f}%")
    print(f" Mean Value   : {mean_val:.2f}%")
    print(f" Std Dev      : {std_val:.2f}%")
    print("=" * 65 + "\n")

    return summary


def export_es_dataset(
    df: pd.DataFrame,
    entity_id: str = ES_ENTITY_ID,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Export the ingested minute-level DataFrame to CSV and Excel (.xlsx) files.

    Returns
    -------
    tuple of (Path to CSV file, Path to Excel file)
    """
    target_dir = output_dir or DATA_ES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    DATA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    safe_entity = re.sub(r"[^\w\-]", "_", entity_id)

    csv_path = target_dir / f"es_metrics_{safe_entity}.csv"
    xlsx_path = target_dir / f"es_metrics_{safe_entity}.xlsx"

    # Save CSV to target_dir and DATA_EXPORT_DIR
    df.to_csv(csv_path, index=False)
    if target_dir != DATA_EXPORT_DIR:
        df.to_csv(DATA_EXPORT_DIR / f"es_metrics_{safe_entity}.csv", index=False)
    logger.info("Saved exported metric dataset to CSV: %s", csv_path)

    # Save Excel with summary sheet
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Minute_Metrics", index=False)

        # Add summary stats sheet
        summary = describe_es_dataset(df, entity_id=entity_id)
        df_summary = pd.DataFrame([summary])
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    logger.info("Saved exported metric dataset to Excel: %s", xlsx_path)

    return csv_path, xlsx_path
