# Infrastructure Metrics Forecasting & Capacity Planning Pipeline

Production-grade time-series forecasting across three distinct modeling architectures and two data pipelines:

1. **Models**:
   - **Amazon Chronos-2**: Universal 120M-parameter pretrained transformer (`amazon/chronos-2`) supporting native **Multivariate Forecasting** via Group Attention and zero-shot cross-learning.
   - **Google TimesFM 3.0**: Pretrained 330M-parameter foundation model (`google/timesfm-3.0-pytorch`) performing zero-shot multi-step forecasting with quantile uncertainty bounds.
   - **NeuralProphet**: Deep-learning AR neural network with lagged cross-metric regressors and Huber loss.
   - **Holt-Winters Exponential Smoothing**: Additive level, trend, and diurnal/weekly seasonal smoothing with 95% confidence intervals.
2. **Pipelines**:
   - **CSV / Excel Server Metrics**: Dynatrace host metrics (CPU %, Memory %, Disk %, Disk Read/Write bytes) for servers **HYDUPINTAPP16** and **JPRUPIWEBCRP02**.
   - **Elasticsearch Live Metrics**: Stream of minute-level infrastructure metrics (`meter_vm_cpu_total_percentage`) normalized across CPU cores, resampled to 1-minute, hourly (`1h`), or daily (`D`) resolutions.

---

## Clean Segregated Architecture

All models, data exports, forecasts, and evaluations are strictly segregated by model and data pipeline:

```
Phrophet_forecast/
├── config/
│   └── settings.py               ← Central configurations, directory paths & hyperparameters
├── pipeline/
│   ├── ingestion/                ← Segregated data ingestion layer
│   │   ├── csv_ingestion.py      ← Dynatrace Excel/CSV parsing, host alignment & validation
│   │   └── es_ingestion.py       ← Elasticsearch live query, core normalization & resampling
│   ├── models/                   ← Segregated forecasting models
│   │   ├── chronos/              ← Amazon Chronos-2 Universal Model (Multivariate Group Attention)
│   │   │   ├── model_loader.py   ← Device resolver ('auto', 'cpu', 'cuda') & singleton loader
│   │   │   ├── chronos_csv.py    ← Chronos-2 multi-metric CSV forecaster & clipper
│   │   │   └── chronos_es.py     ← Chronos-2 Elasticsearch stream & 30d capacity forecaster
│   │   ├── timesfm/              ← Google TimesFM 3.0 Foundation Model
│   │   │   ├── model_loader.py   ← Device resolver & singleton loader
│   │   │   ├── tfm_csv.py        ← TimesFM multi-metric CSV forecaster & clipper
│   │   │   └── tfm_es.py         ← TimesFM Elasticsearch stream & capacity forecaster
│   │   ├── neuralprophet/
│   │   │   ├── np_csv.py         ← NeuralProphet multi-metric CSV forecaster (two-stage regressors)
│   │   │   └── np_es.py          ← NeuralProphet Elasticsearch stream forecaster
│   │   └── holt_winters/
│   │       ├── hw_csv.py         ← Holt-Winters CSV multi-metric forecaster
│   │       └── hw_es.py          ← Holt-Winters Elasticsearch stream forecaster
│   └── common/                   ← Shared evaluation, benchmark & visualization utilities
│       ├── evaluation.py         ← MAE, RMSE, WAPE, MAPE metrics & holdout split
│       ├── comparison.py         ← Side-by-side model benchmarking & evaluation export
│       └── visualization.py      ← High-resolution dark-themed forecasting charts
├── cli/                          ← Modular CLI subcommand routers
│   ├── cli_chronos.py            ← Router for 'chronos {csv, es}' commands (Universal / Multivariate)
│   ├── cli_timesfm.py            ← Router for 'timesfm {csv, es}' commands (CPU/GPU ready)
│   ├── cli_neuralprophet.py      ← Router for 'np {csv, es}' commands
│   ├── cli_holtwinters.py        ← Router for 'hw {csv, es}' commands
│   └── cli_compare.py           ← Router for 'compare {csv, es}' benchmarking
├── models/saved/                 ← Segregated model checkpoints & metadata
│   ├── chronos/                  ← Chronos-2 metadata & configuration for hosts and ES
│   ├── timesfm/                  ← TimesFM metadata & configuration for hosts and ES
│   ├── neuralprophet/            ← Trained .np checkpoints & PyTorch .pt models
│   └── holt_winters/             ← Trained .joblib models for host and ES metrics
├── outputs/
│   ├── data/                     ← Exported datasets for verification
│   ├── forecasts/                ← Segregated forecast CSVs, Excel workbooks, and PNG charts
│   │   ├── chronos/              ← Chronos-2 host metric charts, CSVs, and 30d capacity charts
│   │   ├── timesfm/              ← TimesFM host metric charts, CSVs, and capacity exports
│   │   ├── neuralprophet/        ← NeuralProphet host metric charts & CSVs
│   │   └── holt_winters/         ← Holt-Winters host metric charts & CSVs
│   └── evaluations/              ← Benchmark comparison reports
├── tests/                        ← Comprehensive unit test suite (53 tests passing)
│   ├── test_chronos.py           ← Chronos-2 multivariate, device resolution & capacity tests
│   ├── test_timesfm.py           ← TimesFM device resolution, predict & evaluate tests
│   ├── test_holt_winters.py      ← Holt-Winters unit tests
│   ├── test_forecaster.py        ← NeuralProphet unit tests
│   └── test_ingestion.py         ← Ingestion validation tests
├── main.py                       ← Unified master CLI (chronos, timesfm, np, hw, compare)
├── main_chronos.py               ← Dedicated Amazon Chronos-2 CLI
├── main_tfm.py                   ← Dedicated Google TimesFM 3.0 CLI
├── main_np.py                    ← Dedicated NeuralProphet CLI
├── main_hw.py                    ← Dedicated Holt-Winters CLI
├── main_es.py                    ← Dedicated Elasticsearch legacy CLI
└── requirements.txt              ← Dependencies
```

---

## Amazon Chronos-2 Foundation Model

Amazon's **Chronos-2** (`amazon/chronos-2`) is a 120M-parameter T5-encoder universal forecasting model capable of zero-shot predictions on univariate, multivariate, and covariate-rich time series.

### Key Highlights:
- **Multivariate Group Attention**: Inter-series cross-learning across all host telemetry (CPU, Memory, Disk, Network) processed concurrently in a single forward pass.
- **Univariate Toggle**: Support for `--univariate` to isolate individual series without cross-metric attention.
- **Device Agnostic (CPU & GPU)**: Runs on local CPU out-of-the-box and auto-migrates to CUDA GPU when transferred to a GPU server.
- **30-Day Capacity Planning**: Hourly projection curve with threshold warnings + Daily summary peak/mean bar chart.

---

## Google TimesFM 3.0 Foundation Model

Google's **TimesFM 3.0** (`google/timesfm-3.0-pytorch`) is a 330M-parameter decoder-only foundation transformer trained on over 1 trillion time points.

---

## Command-Line Interface

### 1. Amazon Chronos-2 (`main_chronos.py` or `main.py chronos`)

```bash
# --- CSV Host Pipeline ---
# 1. Generate 30-day forecast CSV, Excel, and dark-themed PNG chart (Multivariate default)
python main_chronos.py csv predict --host HYDUPINTAPP16 --periods 30 --device auto

# 2. Run in Univariate mode (disable cross-metric learning)
python main_chronos.py csv predict --host HYDUPINTAPP16 --periods 30 --univariate

# 3. Evaluate accuracy on a 14-day trailing holdout window
python main_chronos.py csv evaluate --host HYDUPINTAPP16 --holdout 14 --device auto

# --- Elasticsearch Pipeline ---
# 1. Predict 24 hours ahead on 1h resolution
python main_chronos.py es predict --resolution 1h --periods 24 --device auto

# 2. Evaluate 24-hour holdout on 1h resolution
python main_chronos.py es evaluate --resolution 1h --holdout 24 --device auto

# 3. Run 30-day capacity planning (exports hourly & daily CSV/Excel and PNG charts)
python main_chronos.py es capacity --forecast-days 30 --device auto
```

### 2. Google TimesFM 3.0 (`main_tfm.py` or `main.py timesfm`)

```bash
# --- CSV Host Pipeline ---
# 1. Warm up model & verify host schema
python main_tfm.py csv fit --host HYDUPINTAPP16 --device auto

# 2. Generate 30-day forecast CSV and PNG charts
python main_tfm.py csv predict --host HYDUPINTAPP16 --periods 30 --device auto

# 3. Evaluate accuracy on a 14-day trailing holdout window
python main_tfm.py csv evaluate --host HYDUPINTAPP16 --holdout 14 --device auto

# --- Elasticsearch Pipeline ---
# 1. Predict 24 hours ahead on 1h resolution
python main_tfm.py es predict --resolution 1h --periods 24 --device auto

# 2. Evaluate 24-hour holdout on 1h resolution
python main_tfm.py es evaluate --resolution 1h --holdout 24 --device auto

# 3. Run 30-day capacity planning (exports hourly & daily CSV/Excel)
python main_tfm.py es capacity --forecast-days 30 --device auto
```

### 3. Holt-Winters Commands (`main_hw.py` or `main.py hw`)
```bash
# CSV Pipeline: Fit, predict, evaluate
python main_hw.py csv fit --host HYDUPINTAPP16
python main_hw.py csv predict --host HYDUPINTAPP16 --periods 30
python main_hw.py csv evaluate --host HYDUPINTAPP16 --holdout 14

# Elasticsearch Pipeline
python main_hw.py es fit --resolution 1h
python main_hw.py es predict --resolution 1h --periods 24
python main_hw.py es evaluate --resolution 1h --holdout 24
python main_hw.py es capacity --forecast-days 30
```

### 4. NeuralProphet Commands (`main_np.py` or `main.py np`)
```bash
# CSV Pipeline
python main_np.py csv fit --host HYDUPINTAPP16 --n-lags 7
python main_np.py csv predict --host HYDUPINTAPP16 --periods 30
python main_np.py csv evaluate --host HYDUPINTAPP16 --holdout 30

# Elasticsearch Pipeline
python main_np.py es fit --resolution 1h --epochs 50
python main_np.py es predict --resolution 1h --periods 24
python main_np.py es evaluate --resolution 1h --holdout 24
```

---

## GPU Migration Guide

When transferring this project to a GPU server:

1. **Transfer the project folder**:
   ```bash
   scp -r Phrophet_forecast/ user@gpu-server:/path/to/Phrophet_forecast/
   ```
2. **Install GPU PyTorch & Dependencies**:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. **Run on GPU**:
   The code automatically detects `torch.cuda.is_available()`:
   ```bash
   # Automatically runs on GPU (or explicitly pass --device cuda)
   python main_tfm.py csv predict --host HYDUPINTAPP16 --device auto
   python main_tfm.py es predict --resolution 1h --periods 24 --device cuda
   ```
   **No code modifications are required.**

---

## Empirical Benchmark Findings

### 1. CSV Host Metrics Evaluation (HYDUPINTAPP16, 14-Day Holdout)
| Metric | NeuralProphet RMSE | Holt-Winters RMSE | TimesFM 3.0 RMSE | TimesFM 3.0 WAPE |
|---|---|---|---|---|
| **CPU Usage (%)** | 4.73 pp | **1.31 pp** | 1.70 pp | 74.63% |
| **Memory Available (%)** | **2.83 pp** | 3.72 pp | 4.83 pp | **5.13%** |
| **Disk Available (%)** | 11.44 pp | **0.86 pp** | 1.02 pp | **1.14%** |
| **Disk Read (bytes/s)** | 86.07 KB/s | **2.97 KB/s** | 5.58 KB/s | **20.86%** |
| **Disk Write (bytes/s)** | 24.32 KB/s | **3.58 KB/s** | 4.79 KB/s | **47.50%** |

### 2. Elasticsearch Metrics (1-Hour Resolution, 24-Hour Holdout)
| Model | MAE | RMSE | WAPE | Training / Fit Time |
|---|---|---|---|---|
| **NeuralProphet** | 33.23 pp | 42.20 pp | N/A | ~45s |
| **Holt-Winters** | **1.15 pp** | **1.42 pp** | **9.56%** | ~0.05s |
| **Google TimesFM 3.0** | 1.33 pp | 3.77 pp | 10.37% | **0s (Zero-Shot)** |

---

## Testing & Quality Assurance

Run the automated test suite covering all models, device resolution, data ingestion, boundary clipping, and evaluations:

```bash
.venv/bin/pytest tests/ -v
```

All 43 unit and integration tests pass cleanly.
