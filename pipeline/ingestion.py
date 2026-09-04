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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    CPU_FILE,
    DISK_FILE,
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
) -> dict[str, pd.DataFrame]:
    """
    Ingest the three Dynatrace Excel exports and return one DataFrame per host.

    Parameters
    ----------
    cpu_file  : Path to CPU usage % Excel file.
    mem_file  : Path to Memory available % Excel file.
    disk_file : Path to Disk available % Excel file.

    Returns
    -------
    dict mapping short host alias → pd.DataFrame with columns:
        - ``ds``         : datetime64[ns], daily frequency (midnight-normalised)
        - ``cpu_pct``    : float, CPU usage in percent (0–100)
        - ``memory_pct`` : float, Memory available in percent (0–100)
        - ``disk_pct``   : float, Disk available in percent (0–100)

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

    result: dict[str, pd.DataFrame] = {}

    for alias, col_name in HOSTS.items():
        logger.info("Preprocessing host: %s  (column='%s')", alias, col_name)
        df = _build_host_dataframe(alias, col_name, cpu_raw, mem_raw, disk_raw)
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
) -> pd.DataFrame:
    """
    Extract one host's time series from the three raw DataFrames.

    Column-order independence
    -------------------------
    Each metric file exposes the host data via a *named* column lookup
    (``raw_df[col_name]``), not by position.  This is the defensive fix for
    the known Memory-file column order swap where the two hosts appear in
    reversed order compared to CPU and Disk files.

    Timestamp normalisation
    -----------------------
    Raw ``Date`` values carry a 05:30:00 time component (IST export offset).
    We strip this to midnight (``normalize()``) so NeuralProphet receives
    clean daily timestamps without time-zone artifacts.

    Scaling
    -------
    Raw values are decimal proportions in [0, 1].  Multiplying by 100 converts
    them to intuitive percentage values in [0, 100], aligning with the
    ``normalize="minmax"`` setting in NeuralProphet which then maps the
    already-bounded [0, 100] range to [0, 1] internally for stable training.
    """
    # --- Validate that the expected host column exists in every file ---
    for label, raw in [("CPU", cpu_raw), ("Memory", mem_raw), ("Disk", disk_raw)]:
        if col_name not in raw.columns:
            raise KeyError(
                f"Host column '{col_name}' not found in {label} file.\n"
                f"Available columns: {list(raw.columns)}"
            )

    # --- Assemble and scale ---
    df = pd.DataFrame()
    df["ds"] = pd.to_datetime(cpu_raw["Date"]).dt.normalize()  # strip 05:30 offset
    df["cpu_pct"] = cpu_raw[col_name].values * 100.0
    df["memory_pct"] = mem_raw[col_name].values * 100.0
    df["disk_pct"] = disk_raw[col_name].values * 100.0

    # --- Sort by date (defensive; source files appear already sorted) ---
    df = df.sort_values("ds").reset_index(drop=True)

    return df


def _validate(df: pd.DataFrame, alias: str) -> None:
    """
    Run four quality gates on a processed host DataFrame.  Raises on failure.

    Gates
    -----
    1. No NaN values in any column.
    2. Strictly monotonic, contiguous dates (max gap = 1 day).
    3. All percentage values lie in [0, 100].
    4. Minimum row count (≥ 30) to support holdout evaluation.
    """
    # Gate 1: NaN check
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        raise ValueError(
            f"[{alias}] NaN values detected after preprocessing:\n{nan_counts}"
        )

    # Gate 2: Date continuity
    date_diffs = df["ds"].diff().dropna()
    bad_gaps = date_diffs[date_diffs > pd.Timedelta(days=1)]
    if not bad_gaps.empty:
        raise ValueError(
            f"[{alias}] Temporal gaps detected (expected 1-day steps):\n"
            f"{bad_gaps}"
        )

    # Gate 3: Value range
    for col in METRICS:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min < -0.01 or col_max > 100.01:  # small tolerance for float
                raise ValueError(
                    f"[{alias}] Column '{col}' has out-of-range values: "
                    f"min={col_min:.4f}, max={col_max:.4f}.  Expected [0, 100]."
                )

    # Gate 4: Minimum length
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
                print(
                    f"    {col:15s}: "
                    f"min={df[col].min():6.2f}%  "
                    f"max={df[col].max():6.2f}%  "
                    f"mean={df[col].mean():6.2f}%  "
                    f"std={df[col].std():5.2f}%"
                )
    print(f"{'='*60}\n")
