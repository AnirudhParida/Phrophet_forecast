"""
tests/test_ingestion.py
=======================
Unit tests for pipeline/ingestion.py.

Tests do NOT require NeuralProphet to be installed — they only exercise the
data loading and preprocessing logic using synthetic DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.ingestion import (
    _build_host_dataframe,
    _validate,
    describe_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HOST1_COL = "10.50.98.26 - HYDUPINTAPP16"
HOST2_COL = "DR - 10.78.33.83 - JPRUPIWEBCRP02"

N_ROWS = 60  # Enough for holdout + AR warmup


def _make_raw(n_rows: int = N_ROWS, swap_host_order: bool = False) -> dict:
    """Create synthetic raw DataFrames mimicking the real Excel structure."""
    dates = pd.date_range("2025-08-28 05:30:00", periods=n_rows, freq="D")

    cpu_raw = pd.DataFrame({
        "Date": dates,
        HOST1_COL: np.random.uniform(0.007, 0.10, n_rows),
        HOST2_COL: np.random.uniform(0.012, 0.08, n_rows),
    })

    # Memory file has SWAPPED column order — the key bug this pipeline fixes
    if swap_host_order:
        mem_raw = pd.DataFrame({
            "Date": dates,
            HOST2_COL: np.random.uniform(0.67, 0.89, n_rows),  # swapped first
            HOST1_COL: np.random.uniform(0.67, 0.89, n_rows),
        })
    else:
        mem_raw = pd.DataFrame({
            "Date": dates,
            HOST1_COL: np.random.uniform(0.67, 0.89, n_rows),
            HOST2_COL: np.random.uniform(0.67, 0.89, n_rows),
        })

    disk_raw = pd.DataFrame({
        "Date": dates,
        HOST1_COL: np.random.uniform(0.65, 0.76, n_rows),
        HOST2_COL: np.random.uniform(0.62, 0.70, n_rows),
    })

    disk_read_raw = pd.DataFrame({
        "Date": dates,
        HOST1_COL: [f"{val:.1f}kiB" for val in np.random.uniform(10, 100, n_rows)],
        HOST2_COL: [f"{val:.1f}kiB" for val in np.random.uniform(15, 80, n_rows)],
    })

    disk_write_raw = pd.DataFrame({
        "Date": dates,
        HOST1_COL: [f"{val:.1f}kiB" for val in np.random.uniform(20, 200, n_rows)],
        HOST2_COL: [f"{val:.1f}kiB" for val in np.random.uniform(25, 180, n_rows)],
    })

    return {
        "cpu": cpu_raw,
        "mem": mem_raw,
        "disk": disk_raw,
        "disk_read": disk_read_raw,
        "disk_write": disk_write_raw,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildHostDataframe:
    """Tests for _build_host_dataframe()."""

    def test_column_shape(self):
        """Output DataFrame must have all 6 columns: ds, cpu_pct, memory_pct, disk_pct, disk_read_bytes, disk_write_bytes."""
        raw = _make_raw()
        df = _build_host_dataframe(
            "HYDUPINTAPP16", HOST1_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        assert list(df.columns) == [
            "ds", "cpu_pct", "memory_pct", "disk_pct", "disk_read_bytes", "disk_write_bytes"
        ]
        assert len(df) == N_ROWS

    def test_scale_0_to_100(self):
        """All percentage columns must lie in [0, 100] after ×100 scaling."""
        raw = _make_raw()
        df = _build_host_dataframe(
            "HYDUPINTAPP16", HOST1_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        for col in ["cpu_pct", "memory_pct", "disk_pct"]:
            assert df[col].min() >= 0.0, f"{col} has values below 0"
            assert df[col].max() <= 100.0, f"{col} has values above 100"

    def test_column_order_independence(self):
        """
        CRITICAL: Memory file has swapped host column order.
        Name-based lookup must produce DIFFERENT values for HOST1 vs HOST2
        even when memory columns are reversed.
        """
        raw = _make_raw(swap_host_order=True)
        df_host1 = _build_host_dataframe(
            "H1", HOST1_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        df_host2 = _build_host_dataframe(
            "H2", HOST2_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        # memory_pct must differ between hosts (they were populated from different columns)
        assert not np.allclose(
            df_host1["memory_pct"].values,
            df_host2["memory_pct"].values,
        ), "Column order fix failed — both hosts have identical memory values!"

    def test_timestamp_normalized_to_midnight(self):
        """ds column must be midnight timestamps (no 05:30 offset)."""
        raw = _make_raw()
        df = _build_host_dataframe(
            "HYDUPINTAPP16", HOST1_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        for ts in df["ds"]:
            assert ts.hour == 0 and ts.minute == 0, f"Timestamp not normalised: {ts}"

    def test_missing_host_column_raises_key_error(self):
        """KeyError if the host column name is absent from any file."""
        raw = _make_raw()
        with pytest.raises(KeyError, match="not found"):
            _build_host_dataframe(
                "BAD", "NonExistentHost",
                raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
            )

    def test_sorted_ascending_by_date(self):
        """Output must be sorted chronologically."""
        raw = _make_raw()
        # Shuffle the raw CPU dates to test sort
        raw["cpu"] = raw["cpu"].sample(frac=1, random_state=42).reset_index(drop=True)
        df = _build_host_dataframe(
            "HYDUPINTAPP16", HOST1_COL,
            raw["cpu"], raw["mem"], raw["disk"], raw["disk_read"], raw["disk_write"]
        )
        assert df["ds"].is_monotonic_increasing


class TestValidate:
    """Tests for the _validate() quality gate function."""

    def _valid_df(self, n_rows: int = N_ROWS) -> pd.DataFrame:
        dates = pd.date_range("2025-08-28", periods=n_rows, freq="D")
        return pd.DataFrame({
            "ds": dates,
            "cpu_pct": np.random.uniform(1, 10, n_rows),
            "memory_pct": np.random.uniform(70, 90, n_rows),
            "disk_pct": np.random.uniform(65, 76, n_rows),
            "disk_read_bytes": np.random.uniform(100, 1000, n_rows),
            "disk_write_bytes": np.random.uniform(200, 2000, n_rows),
        })

    def test_valid_dataframe_passes(self):
        """A clean DataFrame should pass all gates without raising."""
        df = self._valid_df()
        _validate(df, "TEST_HOST")  # should not raise

    def test_nan_gate(self):
        """NaN values in any column must raise ValueError."""
        df = self._valid_df()
        df.loc[5, "cpu_pct"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            _validate(df, "TEST_HOST")

    def test_temporal_gap_gate(self, caplog):
        """A date gap > 1 day must trigger a warning log."""
        df = self._valid_df()
        df.at[10, "ds"] = df.at[9, "ds"] + pd.Timedelta(days=3)
        df = df.sort_values("ds").reset_index(drop=True)
        with caplog.at_level("WARNING"):
            _validate(df, "TEST_HOST")
        assert "Temporal gaps detected" in caplog.text

    def test_out_of_range_gate(self):
        """Values > 100 must raise ValueError."""
        df = self._valid_df()
        df.loc[0, "cpu_pct"] = 110.0
        with pytest.raises(ValueError, match="out-of-range"):
            _validate(df, "TEST_HOST")

    def test_minimum_length_gate(self):
        """DataFrames with fewer than 30 rows must raise ValueError."""
        df = self._valid_df(n_rows=10)
        with pytest.raises(ValueError, match="Insufficient"):
            _validate(df, "TEST_HOST")
