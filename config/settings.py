"""
config/settings.py
==================
Centralised configuration for the NeuralProphet infrastructure forecasting
pipeline.  All magic strings, file paths, and hyper-parameters live here so
that no value is hard-coded anywhere else in the codebase.

Changing a host name, data path, or training epoch count requires editing
only this single file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the Phrophet_forecast project (one directory above this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Raw Excel files (Dynatrace exports)
DATA_DIR: Path = Path(
    "/home/anirudh.parida@apmosys.mahape/Documents/hostmetricdata"
)

CPU_FILE: Path = (
    DATA_DIR
    / "CPU usage % +2 (Aug 29, 2025, 05_30 - Aug 29, 2026, 05_30).xlsx"
)
MEMORY_FILE: Path = (
    DATA_DIR
    / "Memory available % +2 (Aug 29, 2025, 05_30 - Aug 29, 2026, 05_30).xlsx"
)
DISK_FILE: Path = (
    DATA_DIR
    / "Disk available % (Aug 29, 2025, 05_30 - Aug 29, 2026, 05_30).xlsx"
)
DISK_READ_FILE: Path = (
    DATA_DIR
    / "Disk read bytes per second +3 (Aug 29, 2025, 05_30 - Aug 29, 2026, 05_30).xlsx"
)
DISK_WRITE_FILE: Path = (
    DATA_DIR
    / "Disk write bytes per second +3 (Aug 29, 2025, 05_30 - Aug 29, 2026, 05_30).xlsx"
)

# Output directories created automatically at runtime
MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"
FORECASTS_DIR: Path = PROJECT_ROOT / "outputs" / "forecasts"

# ---------------------------------------------------------------------------
# Host Registry
# ---------------------------------------------------------------------------
# Keys  → short, filesystem-safe alias used in filenames and CLI flags.
# Values → exact column header in the Excel files (used for name-based lookup,
#          which is immune to column-order inconsistencies across files).

HOSTS: dict[str, str] = {
    "HYDUPINTAPP16": "10.50.98.26 - HYDUPINTAPP16",
    "JPRUPIWEBCRP02": "DR - 10.78.33.83 - JPRUPIWEBCRP02",
}

# Metrics produced by the ingestion layer
METRICS: list[str] = ["cpu_pct", "memory_pct", "disk_pct", "disk_read_bytes", "disk_write_bytes"]

# Human-readable labels used in charts and log output
METRIC_LABELS: dict[str, str] = {
    "cpu_pct": "CPU Usage (%)",
    "memory_pct": "Memory Available (%)",
    "disk_pct": "Disk Available (%)",
    "disk_read_bytes": "Disk Read (bytes/s)",
    "disk_write_bytes": "Disk Write (bytes/s)",
}

# ---------------------------------------------------------------------------
# NeuralProphet Hyper-parameters
# ---------------------------------------------------------------------------

# Sampling frequency for NeuralProphet — daily data
FREQ: str = "D"

# n_lags candidates for the compare-lags experiment.
# n_lags=7 : 1-week AR lookback → discards only 1.9 % of 366 rows as warmup.
# n_lags=14: 2-week AR lookback → discards 3.8 % of 366 rows as warmup.
# Both are tested; the one with lower holdout RMSE is recommended.
N_LAGS_OPTIONS: list[int] = [7, 14]

# Default n_lags used when the user does not run the comparison experiment.
N_LAGS_DEFAULT: int = 7

# Per-metric optimal lookback windows derived from empirical lag-comparison experiments:
# - cpu_pct    : 7-day lookback (captures weekly workweek patterns without blurring spikes)
# - memory_pct : 14-day lookback (captures 2-week memory consumption trends)
# - disk_pct   : 14-day lookback (reduces RMSE by ~42% from 5.95 to 3.41 pp)
# - disk_read_bytes : Default 14-day lookback
# - disk_write_bytes : Default 14-day lookback
METRIC_N_LAGS: dict[str, int] = {
    "cpu_pct": 7,
    "memory_pct": 14,
    "disk_pct": 14,
    "disk_read_bytes": 14,
    "disk_write_bytes": 14,
}

# Forecast horizon: 90 calendar days ahead
N_FORECASTS: int = 90

# Huber loss is robust to sudden operational spikes (e.g., batch jobs)
# which would inflate MSE / MAE disproportionately.
LOSS_FUNC: str = "Huber"

# Weekly seasonality captures Mon–Fri workload vs. weekend patterns.
WEEKLY_SEASONALITY: bool = True

# Yearly seasonality disabled: only 1 full year of data is available.
# Enabling it would fit Fourier components to a single seasonal cycle,
# which is high-risk overfitting on 366 observations.
YEARLY_SEASONALITY: bool = False

# Daily (intraday) seasonality is irrelevant — data is already daily-aggregated.
DAILY_SEASONALITY: bool = False

# Training epochs. 200 provides a good loss-convergence balance on ~350 rows.
EPOCHS: int = 200

# Learning rate for Adam optimiser.
LEARNING_RATE: float = 1e-3

# Batch size.  64 is the NeuralProphet default and works well for ~350 rows.
BATCH_SIZE: int = 64

# NeuralProphet internal normalisation: "minmax" maps [0, 100] inputs to
# [0, 1] for numerically stable gradient updates.
NORMALIZE: str = "minmax"

# Quantiles for forecast confidence intervals (5th–95th percentile band)
QUANTILES: list[float] = [0.05, 0.95]

# ---------------------------------------------------------------------------
# Production Enhancements
# ---------------------------------------------------------------------------

# Growth curve type: "linear" or "off" (NeuralProphet trend component)
GROWTH: str = "linear"
CAP_VALUE: float = 100.0
FLOOR_VALUE: float = 0.0

# Non-linear deep AR layers (AR-Net multi-layer perceptron architecture)
# [32, 16] creates a 2-hidden-layer deep AR network to model complex non-linear spikes.
AR_LAYERS: list[int] | None = [32, 16]

# Country holidays integration (Indian national holidays for regional server load patterns)
COUNTRY_HOLIDAYS: str | None = "IN"

# Two-stage forecast regressor pipeline: use stage-1 forecasts for future regressors
TWO_STAGE_REGRESSORS: bool = True

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Number of days to hold out from training for model evaluation
HOLDOUT_DAYS: int = 30

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------

# Maximum worker processes for parallel host fitting
MAX_WORKERS: int = 2   # one per host; increase if more hosts are added

