"""
config/settings.py
==================
Centralised configuration for the NeuralProphet infrastructure forecasting
pipeline.  All magic strings, file paths, and hyper-parameters live here so
that no value is hard-coded anywhere else in the codebase.

Changing a host name, data path, or training epoch count requires editing
only this single file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

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

# ---------------------------------------------------------------------------
# Segregated Output & Model Directory Hierarchy
# ---------------------------------------------------------------------------

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved"

# Data exports: outputs/data/{csv, es}/
DATA_EXPORT_DIR: Path = OUTPUTS_DIR / "data"
DATA_CSV_DIR: Path = DATA_EXPORT_DIR / "csv"
DATA_ES_DIR: Path = DATA_EXPORT_DIR / "es"

# Model checkpoints: models/saved/{neuralprophet, holt_winters}/{csv, es}/
NP_MODELS_DIR: Path = MODELS_DIR / "neuralprophet"
NP_CSV_MODELS_DIR: Path = NP_MODELS_DIR / "csv"
NP_ES_MODELS_DIR: Path = NP_MODELS_DIR / "es"

HW_MODELS_DIR_BASE: Path = MODELS_DIR / "holt_winters"
HW_CSV_MODELS_DIR: Path = HW_MODELS_DIR_BASE / "csv"
HW_ES_MODELS_DIR: Path = HW_MODELS_DIR_BASE / "es"

TFM_MODELS_DIR: Path = MODELS_DIR / "timesfm"
TFM_CSV_MODELS_DIR: Path = TFM_MODELS_DIR / "csv"
TFM_ES_MODELS_DIR: Path = TFM_MODELS_DIR / "es"

CHRONOS_MODELS_DIR: Path = MODELS_DIR / "chronos"
CHRONOS_CSV_MODELS_DIR: Path = CHRONOS_MODELS_DIR / "csv"
CHRONOS_ES_MODELS_DIR: Path = CHRONOS_MODELS_DIR / "es"

# Forecast outputs: outputs/forecasts/{neuralprophet, holt_winters, timesfm, chronos}/{csv, es}/
FORECASTS_DIR_BASE: Path = OUTPUTS_DIR / "forecasts"
FORECASTS_NP_CSV_DIR: Path = FORECASTS_DIR_BASE / "neuralprophet" / "csv"
FORECASTS_NP_ES_DIR: Path = FORECASTS_DIR_BASE / "neuralprophet" / "es"
FORECASTS_HW_CSV_DIR: Path = FORECASTS_DIR_BASE / "holt_winters" / "csv"
FORECASTS_HW_ES_DIR: Path = FORECASTS_DIR_BASE / "holt_winters" / "es"
FORECASTS_TFM_CSV_DIR: Path = FORECASTS_DIR_BASE / "timesfm" / "csv"
FORECASTS_TFM_ES_DIR: Path = FORECASTS_DIR_BASE / "timesfm" / "es"
FORECASTS_CHRONOS_CSV_DIR: Path = FORECASTS_DIR_BASE / "chronos" / "csv"
FORECASTS_CHRONOS_ES_DIR: Path = FORECASTS_DIR_BASE / "chronos" / "es"

# Evaluation reports: outputs/evaluations/{csv, es}/
EVALUATIONS_DIR: Path = OUTPUTS_DIR / "evaluations"
EVALUATIONS_CSV_DIR: Path = EVALUATIONS_DIR / "csv"
EVALUATIONS_ES_DIR: Path = EVALUATIONS_DIR / "es"

# Backward compatibility aliases
FORECASTS_DIR: Path = FORECASTS_NP_CSV_DIR
ES_MODELS_DIR: Path = NP_ES_MODELS_DIR
ES_FORECASTS_DIR: Path = FORECASTS_NP_ES_DIR
HW_MODELS_DIR: Path = HW_CSV_MODELS_DIR
HW_FORECASTS_DIR: Path = FORECASTS_HW_CSV_DIR
HW_ES_FORECASTS_DIR: Path = FORECASTS_HW_ES_DIR
HW_CSV_FORECASTS_DIR: Path = FORECASTS_HW_CSV_DIR
NP_CSV_FORECASTS_DIR: Path = FORECASTS_NP_CSV_DIR
NP_ES_FORECASTS_DIR: Path = FORECASTS_NP_ES_DIR
FORECASTS_TFM_DIR: Path = FORECASTS_TFM_CSV_DIR
TFM_CSV_FORECASTS_DIR: Path = FORECASTS_TFM_CSV_DIR
TFM_ES_FORECASTS_DIR: Path = FORECASTS_TFM_ES_DIR
FORECASTS_CHRONOS_DIR: Path = FORECASTS_CHRONOS_CSV_DIR
CHRONOS_CSV_FORECASTS_DIR: Path = FORECASTS_CHRONOS_CSV_DIR
CHRONOS_ES_FORECASTS_DIR: Path = FORECASTS_CHRONOS_ES_DIR

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

# ---------------------------------------------------------------------------
# Elasticsearch Pipeline Configuration
# ---------------------------------------------------------------------------

ES_HOST: str = "http://192.168.12.94:9200"
ES_USER: str = "elastic"
ES_PASSWORD: str = "Netraa@2026"
ES_INDEX: str = "netraa_metrics-all*"
ES_ENTITY_ID: str = "MTkyLjE2OC4xMi44Ng==.1"
ES_METRIC_TABLE: str = "meter_vm_cpu_total_percentage"
ES_LOOKBACK_DAYS: int = 7
ES_CPU_CORES: float = 6.0  # Host CPU core count to normalize multi-core percentage [0, 600] -> [0, 100]


# Minute-level NeuralProphet Hyper-parameters (default: 1min)
FREQ_ES: str = "1min"
N_LAGS_ES: int = 60         # 1-hour lookback (60 minutes)
N_FORECASTS_ES: int = 60    # 1-hour forecast horizon (60 minutes)
EPOCHS_ES: int = 50         # Optimized epoch count for ~10,000 minute rows
BATCH_SIZE_ES: int = 64
LEARNING_RATE_ES: float = 1e-3
HOLDOUT_MINUTES_ES: int = 120  # 2-hour holdout e   valuation window

# Hourly-resolution NeuralProphet Hyper-parameters
# 7 days of 1-minute data -> resampled to ~168 hourly rows
# Note: NeuralProphet requires total_rows >= n_lags + n_forecasts
FREQ_ES_HOURLY: str = "1h"
N_LAGS_ES_HOURLY: int = 24        # 24-hour lookback (1 day)
N_FORECASTS_ES_HOURLY: int = 24   # 24-hour forecast horizon (1 day)
EPOCHS_ES_HOURLY: int = 100       # More epochs; fewer but richer rows
BATCH_SIZE_ES_HOURLY: int = 32    # Smaller batch for ~168 rows
HOLDOUT_HOURS_ES: int = 24        # 24-hour holdout evaluation window

# Daily-resolution NeuralProphet Hyper-parameters
# 7 days of 1-minute data -> resampled to ~7 daily rows
FREQ_ES_DAILY: str = "D"
N_LAGS_ES_DAILY: int = 2         # 2-day lookback
N_FORECASTS_ES_DAILY: int = 3    # 3-day forecast horizon
EPOCHS_ES_DAILY: int = 150
BATCH_SIZE_ES_DAILY: int = 16
HOLDOUT_DAYS_ES: int = 2         # 2-day holdout evaluation window

# ---------------------------------------------------------------------------
# Holt-Winters Exponential Smoothing Configuration
# ---------------------------------------------------------------------------


# Core Holt-Winters components
# Trend: "add" (additive) or "mul" (multiplicative) or None
HW_TREND: str = "add"
# Damped trend: dampens trend projection over extended horizons
HW_DAMPED_TREND: bool = False
# Seasonal component: "add" (additive) or "mul" (multiplicative) or None
HW_SEASONAL: str = "add"

# Seasonality periods (m) for various sampling frequencies:
# Daily: 7 days/week (Mon-Sun weekly cycle)
HW_SEASONAL_PERIODS_DAILY: int = 7
# Hourly: 24 hours/day (intraday diurnal cycle)
HW_SEASONAL_PERIODS_HOURLY: int = 24
# Minutely: 60 minutes/hour (hourly cycle)
HW_SEASONAL_PERIODS_MINUTELY: int = 60

# ---------------------------------------------------------------------------
# Google TimesFM 3.0 Pretrained Model Configuration
# ---------------------------------------------------------------------------

# Official HuggingFace PyTorch checkpoint ID
TIMESFM_MODEL_ID: str = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-3.0-pytorch")

# Hardware device: "auto" (selects "cuda" if GPU is available, else "cpu"), "cpu", or "cuda"
TIMESFM_DEVICE: str = os.getenv("TIMESFM_DEVICE", "auto")

# Hugging Face auth token from .env or environment
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Inference batch size (number of time series processed simultaneously)
TIMESFM_PER_CORE_BATCH_SIZE: int = int(os.getenv("TIMESFM_BATCH_SIZE", "16"))

# Maximum lookback history window fed into TimesFM context (in time steps)
TIMESFM_CONTEXT_LEN: int = int(os.getenv("TIMESFM_CONTEXT_LEN", "512"))

# Forecast horizons (number of steps to predict ahead)
TIMESFM_HORIZON_CSV: int = N_FORECASTS       # 90 days for daily CSV host data
TIMESFM_HORIZON_ES: int = N_FORECASTS_ES     # 60 steps for minute-level ES data
TIMESFM_HORIZON_ES_HOURLY: int = N_FORECASTS_ES_HOURLY  # 24 hours
TIMESFM_HORIZON_ES_DAILY: int = N_FORECASTS_ES_DAILY    # 3 days

# Evaluation holdout periods
TIMESFM_HOLDOUT_DAYS_CSV: int = HOLDOUT_DAYS            # 30 days
TIMESFM_HOLDOUT_MINUTES_ES: int = HOLDOUT_MINUTES_ES    # 120 minutes
TIMESFM_HOLDOUT_HOURS_ES: int = HOLDOUT_HOURS_ES        # 24 hours
TIMESFM_HOLDOUT_DAYS_ES: int = HOLDOUT_DAYS_ES          # 2 days

# ---------------------------------------------------------------------------
# Amazon Chronos-2 Universal Pretrained Model Configuration
# ---------------------------------------------------------------------------

# Official HuggingFace model ID for Chronos-2 (120M parameters, group attention)
CHRONOS_MODEL_ID: str = os.getenv("CHRONOS_MODEL_ID", "amazon/chronos-2")

# Hardware device: "auto" (selects "cuda" if GPU is available, else "cpu"), "cpu", or "cuda"
CHRONOS_DEVICE: str = os.getenv("CHRONOS_DEVICE", "auto")

# Inference batch size (number of time series or groups processed simultaneously)
CHRONOS_BATCH_SIZE: int = int(os.getenv("CHRONOS_BATCH_SIZE", "64"))

# Context length (maximum history sequence fed into Chronos-2)
CHRONOS_CONTEXT_LEN: int = int(os.getenv("CHRONOS_CONTEXT_LEN", "512"))

# Prediction length defaults
CHRONOS_PREDICTION_LENGTH: int = 24
CHRONOS_CAPACITY_HORIZON_DAYS: int = 30

# Forecast horizons
CHRONOS_HORIZON_CSV: int = N_FORECASTS                  # 90 days for daily CSV host data
CHRONOS_HORIZON_ES: int = N_FORECASTS_ES                # 60 steps for minute-level ES data
CHRONOS_HORIZON_ES_HOURLY: int = N_FORECASTS_ES_HOURLY  # 24 hours
CHRONOS_HORIZON_ES_DAILY: int = N_FORECASTS_ES_DAILY    # 3 days

# Evaluation holdout periods
CHRONOS_HOLDOUT_DAYS_CSV: int = HOLDOUT_DAYS            # 30 days
CHRONOS_HOLDOUT_MINUTES_ES: int = HOLDOUT_MINUTES_ES    # 120 minutes
CHRONOS_HOLDOUT_HOURS_ES: int = HOLDOUT_HOURS_ES        # 24 hours
CHRONOS_HOLDOUT_DAYS_ES: int = HOLDOUT_DAYS_ES          # 2 days

