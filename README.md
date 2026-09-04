# NeuralProphet Infrastructure Metrics Forecasting Pipeline

Production-grade daily forecasting of CPU, Memory, and Disk utilisation for
two servers — **HYDUPINTAPP16** and **JPRUPIWEBCRP02** — using
[NeuralProphet](https://neuralprophet.com/) with cross-metric lagged regressors.

---

## Project Structure

```
Phrophet_forecast/
├── config/
│   └── settings.py          ← All constants: paths, host names, hyperparams
├── pipeline/
│   ├── ingestion.py         ← Excel → per-host DataFrames
│   ├── forecaster.py        ← ServerMetricsForecaster (fit / predict / evaluate)
│   ├── evaluation.py        ← MAE, RMSE, MAPE holdout evaluation
│   └── visualization.py     ← Forecast chart generation (PNG)
├── models/saved/            ← Trained .np model files (auto-created)
├── outputs/forecasts/       ← PNG forecast charts (auto-created)
├── tests/
│   ├── test_ingestion.py
│   └── test_evaluation.py
├── main.py                  ← CLI entrypoint
└── requirements.txt         ← Dependencies with install instructions
```

---

## Step-by-Step Setup & Training Guide

Follow these steps **in order**. Each command is run from the project root:
`/home/anirudh.parida@apmosys.mahape/Documents/Phrophet_forecast/`

---

### STEP 1 — Create the virtual environment

```bash
cd /home/anirudh.parida@apmosys.mahape/Documents/Phrophet_forecast
uv venv .venv --python 3.11          # IMPORTANT: must specify 3.11
source .venv/bin/activate
```

> **Why 3.11?** PyTorch does not yet publish wheels for Python 3.13/3.14.
> Running `uv venv` without `--python` picks the latest downloaded Python
> (3.14 on this machine), which makes torch uninstallable. Python 3.11 is
> already cached locally by uv.

---

### STEP 2 — Install PyTorch (CPU build, required by NeuralProphet)

```bash
# Install NeuralProphet — uv auto-resolves the correct torch for your Python + GPU
uv pip install neuralprophet==0.9.0
```

> **GPU detected on this machine (CUDA 13.0).**
> uv resolved `torch==2.14.0+cu130` automatically — no manual torch install needed.
>
> If you want CPU-only (no GPU, smaller install):
> ```bash
> uv pip install neuralprophet==0.9.0 torch \
>   --extra-index-url https://download.pytorch.org/whl/cpu
> ```

---

### STEP 3 — Install all pipeline dependencies

```bash
uv pip install pandas numpy openpyxl scipy scikit-learn \
               matplotlib seaborn tqdm python-dotenv pyyaml pytest
```

---

### STEP 4 — Verify installation

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import neuralprophet; print('neuralprophet:', neuralprophet.__version__)"
```

Expected output (GPU machine):
```
torch: 2.14.0+cu130
neuralprophet: 0.9.0
```

> The `Importing plotly failed` warning is **harmless** — we use matplotlib for charts.

---

### STEP 5 — Inspect the dataset (no training required)

```bash
python main.py inspect
```

Expected output: date range, row count, and per-metric statistics for each host.
This is your sanity check before training — verify the numbers match the Excel data.

---

### STEP 6A — Run the tests (no NeuralProphet training needed)

Confirm the data pipeline and evaluation logic are correct before spending time training:

```bash
python -m pytest tests/ -v
```

All tests should pass. These tests use synthetic data and do not require a trained model.

---

### STEP 7 — Experiment: Compare n_lags=7 vs n_lags=14

This is the recommended first training step. It trains two sets of models with
different AR lookback windows and prints a side-by-side RMSE comparison.
**Training time: ~15–25 minutes per host (CPU).**

```bash
# Run comparison for HYDUPINTAPP16
python main.py compare-lags --host HYDUPINTAPP16 --holdout 30

# Run comparison for JPRUPIWEBCRP02
python main.py compare-lags --host JPRUPIWEBCRP02 --holdout 30
```

The output table shows RMSE for each metric at n_lags=7 and n_lags=14 and marks
the winner.  Use that n_lags value in all subsequent commands.

---

### STEP 8 — Train the final models (use whichever n_lags won)

Replace `--n-lags 7` with `--n-lags 14` if that won the comparison:

```bash
# Train HYDUPINTAPP16 (all 3 metric models)
python main.py fit --host HYDUPINTAPP16 --n-lags 7

# Train JPRUPIWEBCRP02 (all 3 metric models)
python main.py fit --host JPRUPIWEBCRP02 --n-lags 7
```

**Or train both hosts at once (sequentially):**
```bash
python main.py fit --all --n-lags 7
```

**Or train both hosts in parallel (faster if CPU has multiple cores):**
```bash
python main.py fit --all --parallel --n-lags 7
```

Models are saved to `models/saved/` as `.np` files:
```
models/saved/HYDUPINTAPP16_cpu_pct_lags7.np
models/saved/HYDUPINTAPP16_memory_pct_lags7.np
models/saved/HYDUPINTAPP16_disk_pct_lags7.np
models/saved/JPRUPIWEBCRP02_cpu_pct_lags7.np
...
```

---

### STEP 9 — Generate 7-day forecast charts

After training, generate the time-series forecast images:

```bash
python main.py predict --host HYDUPINTAPP16 --n-lags 7
python main.py predict --host JPRUPIWEBCRP02 --n-lags 7
```

Charts saved to `outputs/forecasts/`:
```
outputs/forecasts/HYDUPINTAPP16_cpu_pct_lags7_forecast.png
outputs/forecasts/HYDUPINTAPP16_memory_pct_lags7_forecast.png
outputs/forecasts/HYDUPINTAPP16_disk_pct_lags7_forecast.png
outputs/forecasts/HYDUPINTAPP16_overview_lags7.png        ← all 3 in one image
outputs/forecasts/JPRUPIWEBCRP02_overview_lags7.png
...
```

---

### STEP 10 — Evaluate model accuracy on holdout data

```bash
python main.py evaluate --host HYDUPINTAPP16 --holdout 30 --n-lags 7
python main.py evaluate --host JPRUPIWEBCRP02 --holdout 30 --n-lags 7
```

Output table:
```
============================================================
  Evaluation: HYDUPINTAPP16  |  n_lags=7
============================================================
  Metric          MAE       RMSE       MAPE
  -----------------------------------------------
  cpu_pct        x.xxxx    x.xxxx     x.xx%
  memory_pct     x.xxxx    x.xxxx     x.xx%
  disk_pct       x.xxxx    x.xxxx     x.xx%
============================================================
```

---

## Hyperparameter Reference

| Parameter | Value | Rationale |
|---|---|---|
| `n_forecasts` | 7 | 7-day ahead horizon |
| `n_lags` | 7 or 14 | Compared empirically (STEP 7) |
| `yearly_seasonality` | False | Only 1 year of data → overfitting risk |
| `weekly_seasonality` | True | Mon–Fri vs weekend load differs |
| `daily_seasonality` | False | Data is already daily-aggregated |
| `loss_func` | Huber | Robust to operational spikes |
| `normalize` | minmax | Maps [0,100] → [0,1] for stable gradients |
| `epochs` | 200 | Good convergence on ~336 training rows |
| `batch_size` | 64 | NeuralProphet default |
| `quantiles` | [0.05, 0.95] | 90% confidence intervals in charts |

---

## Data Source

Raw Dynatrace Excel exports at:
```
/home/anirudh.parida@apmosys.mahape/Documents/hostmetricdata/
```

| File | Metric |
|---|---|
| `CPU usage % +2 ...xlsx` | CPU utilisation (decimal → ×100 = %) |
| `Memory available % +2 ...xlsx` | Memory available (decimal → ×100 = %) |
| `Disk available % ...xlsx` | Disk available (decimal → ×100 = %) |

> **Known quirk:** The Memory file has host columns in **reverse order** vs. CPU and Disk.
> The pipeline resolves this automatically via name-based column lookup.
