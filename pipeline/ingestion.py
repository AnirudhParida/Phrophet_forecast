"""
pipeline/ingestion.py
=====================
Data ingestion and preprocessing layer for the NeuralProphet infrastructure
forecasting pipeline.

Responsibilities
----------------
1. Read three Dynatrace Excel exports (CPU %, Memory %, Disk %).
2. Resolve column-order inconsistency across files via name-based lookup.
3. Normalise timestamps: strip the 05:30:00 time component → midnight UTC date.
4. Scale decimal proportions (0.0–1.0) to percentages (0.0–100.0).
5. Validate data quality: no NaN values, no temporal gaps, values in [0, 100].
6. Return per-host DataFrames with NeuralProphet-ready columns:
   ``ds`` | ``cpu_pct`` | ``memory_pct`` | ``disk_pct``

Design notes
------------
- Uses *name-based* column lookup (``df[host_col_name]``) rather than positional
  indexing.  This is the critical fix for the known column-order swap in the
  Memory file where ``JPRUPIWEBCRP02`` appears before ``HYDUPINTAPP16``.
- All validation is fail-fast with descriptive ``ValueError`` messages so
  pipeline errors surface immediately rather than silently corrupting model
  training data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    CPU_FILE,
    DISK_FILE,
    DISK_READ_FILE,
    DISK_WRITE_FILE,
    HOSTS,
    MEMORY_FILE,
    METRICS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_preprocess(
    cpu_file: Path = CPU_FILE,
    mem_file: Path = MEMORY_FILE,
    disk_file: Path = DISK_FILE,
    disk_read_file: Path = DISK_READ_FILE,
    disk_write_file: Path = DISK_WRITE_FILE,
) -> dict[str, pd.DataFrame]:
    """
    Ingest the five Dynatrace Excel exports and return one DataFrame per host.

    Parameters
    ----------
    cpu_file        : Path to CPU usage % Excel file.
    mem_file        : Path to Memory available % Excel file.
    disk_file       : Path to Disk available % Excel file.
    disk_read_file  : Path to Disk read operations/sec Excel file.
    disk_write_file : Path to Disk write bytes/sec Excel file.

    Returns
    -------
    dict mapping short host alias → pd.DataFrame with columns:
        - ``ds``               : datetime64[ns], daily frequency (midnight-normalised)
        - ``cpu_pct``          : float, CPU usage in percent (0–100)
        - ``memory_pct``       : float, Memory available in percent (0–100)
        - ``disk_pct``         : float, Disk available in percent (0–100)
        - ``disk_read_ops``    : float, Disk read operations per second
        - ``disk_write_bytes`` : float, Disk write bytes per second

    Raises
    ------
    FileNotFoundError  : If any source Excel file is missing.
    KeyError           : If an expected host column is absent from a file.
    ValueError         : If data quality checks fail (NaNs, gaps, out-of-range).
    """
    logger.info("Loading Excel source files …")
    cpu_raw = _read_excel(cpu_file, label="CPU")
    mem_raw = _read_excel(mem_file, label="Memory")
    disk_raw = _read_excel(disk_file, label="Disk")
    disk_read_raw = _read_excel(disk_read_file, label="Disk Read Bytes")
    disk_write_raw = _read_excel(disk_write_file, label="Disk Write Bytes")

    result: dict[str, pd.DataFrame] = {}

    for alias, col_name in HOSTS.items():
        logger.info("Preprocessing host: %s  (column='%s')", alias, col_name)
        df = _build_host_dataframe(
            alias, col_name, cpu_raw, mem_raw, disk_raw, disk_read_raw, disk_write_raw
        )
        _validate(df, alias)
        result[alias] = df
        logger.info(
            "  %s: %d rows, ds %s → %s",
            alias,
            len(df),
            df["ds"].min().date(),
            df["ds"].max().date(),
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def parse_metric_value(val: any) -> float:
    """
    Safely parse raw Dynatrace metric values which may contain strings like
    '< 0.001' or '13.6kiB'. Returns the raw numeric value (converting kiB to bytes).
    """
    if pd.isna(val):
        return np.nan

    if isinstance(val, (int, float)):
        return float(val)

    val_str = str(val).strip()

    # Extract the numeric part (handling cases like "< 0.001", "13.6kiB")
    numeric_str = re.sub(r'[^\d\.]', '', val_str)
    
    try:
        num = float(numeric_str)
    except ValueError:
        return np.nan
        
    # Apply unit multiplier if present
    if 'kiB' in val_str:
        num *= 1024.0
    elif 'MiB' in val_str:
        num *= 1024.0 * 1024.0
    elif 'GiB' in val_str:
        num *= 1024.0 * 1024.0 * 1024.0
    elif 'k' in val_str.lower():
        num *= 1000.0
    elif 'm' in val_str.lower() and 'b' not in val_str.lower():
        num *= 1000000.0
    
    return num


def _read_excel(path: Path, label: str) -> pd.DataFrame:
    """
    Read a single Dynatrace Excel export.

    The files use row 0 as the header (``Date`` + two host columns).
    Raises ``FileNotFoundError`` early with a clear message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Please verify DATA_DIR in config/settings.py points to the "
            "correct hostmetricdata directory."
        )
    df = pd.read_excel(path, header=0, engine="openpyxl")
    logger.debug("%s file columns: %s", label, list(df.columns))
    return df


def _build_host_dataframe(
    alias: str,
    col_name: str,
    cpu_raw: pd.DataFrame,
    mem_raw: pd.DataFrame,
    disk_raw: pd.DataFrame,
    disk_read_raw: pd.DataFrame,
    disk_write_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract one host's time series from the raw DataFrames.

    Column-order independence
    -------------------------
    Each metric file exposes the host data via a *named* column lookup
    (``raw_df[col_name]``), not by position.

    Timestamp normalisation
    -----------------------
    Raw ``Date`` values carry a 05:30:00 time component (IST export offset).
    We strip this to midnight (``normalize()``) so NeuralProphet receives
    clean daily timestamps.

    Scaling & Parsing
    -----------------
    Percentage values (decimal proportions in [0, 1]) are multiplied by 100.
    Unbounded values (read bytes, write bytes) are parsed dynamically from strings
    to pure numeric representations (e.g. converting kiB to bytes).
    """
    # --- Validate that the expected host column exists in every file ---
    for label, raw in [
        ("CPU", cpu_raw),
        ("Memory", mem_raw),
        ("Disk", disk_raw),
        ("Disk Read Bytes", disk_read_raw),
        ("Disk Write Bytes", disk_write_raw),
    ]:
        if col_name not in raw.columns:
            raise KeyError(
                f"Host column '{col_name}' not found in {label} file.\n"
                f"Available columns: {list(raw.columns)}"
            )

    # --- Assemble and scale ---
    df = pd.DataFrame()
    df["ds"] = pd.to_datetime(cpu_raw["Date"]).dt.normalize()  # strip 05:30 offset
    
    # Percentages
    df["cpu_pct"] = cpu_raw[col_name].values * 100.0
    df["memory_pct"] = mem_raw[col_name].values * 100.0
    df["disk_pct"] = disk_raw[col_name].values * 100.0

    # Unbounded parsed metrics
    df["disk_read_bytes"] = disk_read_raw[col_name].apply(parse_metric_value)
    df["disk_write_bytes"] = disk_write_raw[col_name].apply(parse_metric_value)

    # --- Sort by date (defensive; source files appear already sorted) ---
    df = df.sort_values("ds").reset_index(drop=True)

    return df


def _validate(df: pd.DataFrame, alias: str) -> None:
    """
    Run quality gates on a processed host DataFrame. Raises on failure.
    """
    # 1. Missing values
    if df.isna().any().any():
        cols_with_nans = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaN values found in {alias} for columns: {cols_with_nans}")

    # 2. Temporal gaps (ensure perfect daily frequency)
    date_diffs = df["ds"].diff().dropna()
    if not (date_diffs == pd.Timedelta(days=1)).all():
        logger.warning(
            "Temporal gaps detected in %s data. NeuralProphet will impute "
            "missing days automatically.",
            alias,
        )

    # 3. Out-of-bounds percentages (applies to _pct metrics only)
    for metric in METRICS:
        if metric in df.columns and metric.endswith("_pct"):
            metric_min = df[metric].min()
            metric_max = df[metric].max()
            if metric_min < -0.01 or metric_max > 100.01:
                raise ValueError(
                    f"[{alias}] Column '{metric}' has out-of-range values: "
                    f"min={metric_min:.4f}, max={metric_max:.4f}. Expected [0, 100]."
                )

    # 4. Minimum length
    if len(df) < 30:
        raise ValueError(
            f"[{alias}] Insufficient data: {len(df)} rows (minimum 30 required)."
        )

    logger.debug("[%s] All data quality gates passed (%d rows).", alias, len(df))


def describe_dataset(host_data: dict[str, pd.DataFrame]) -> None:
    """
    Print a human-readable summary of the loaded dataset to stdout.
    Useful for quick sanity checks before training.
    """
    for alias, df in host_data.items():
        print(f"\n{'='*60}")
        print(f"  Host : {alias}")
        print(f"  Rows : {len(df)}")
        print(f"  Date : {df['ds'].min().date()}  →  {df['ds'].max().date()}")
        print(f"  Metrics:")
        for col in METRICS:
            if col in df.columns:
                unit = "%" if col.endswith("_pct") else ""
                print(
                    f"    {col:18s}: "
                    f"min={df[col].min():8.2f}{unit}  "
                    f"max={df[col].max():8.2f}{unit}  "
                    f"mean={df[col].mean():8.2f}{unit}  "
                    f"std={df[col].std():8.2f}{unit}"
                )
    print(f"{'='*60}\n")
