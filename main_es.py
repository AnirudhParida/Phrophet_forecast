"""
main_es.py
==========
Command-line interface for the Elasticsearch Infrastructure Metrics Forecasting Pipeline.

Available commands
------------------
  inspect    — Fetch metric data from Elasticsearch, display min/max summary stats, and export CSV/Excel files.
  fit        — Train NeuralProphet model on 1-minute resolution ES metrics.
  predict    — Generate N-step minute-level future predictions and plot forecast chart.
  capacity   — [NEW] Train on 1-min data, forecast 30 days, export hourly and daily CSV/Excel + charts.
  evaluate   — Compute MAE / RMSE / MAPE on a trailing holdout window.

Resolution options (for fit/predict/evaluate only)
------------------
  --resolution 1min   1-minute granularity, forecast 60 minutes ahead (default)
  --resolution 1h     Hourly granularity,  forecast 30 days (720 hours) ahead
  --resolution D      Daily granularity,   forecast 30 days ahead

Quick Start — 30-Day Capacity Planning
---------------------------------------
  python main_es.py capacity               # fits on 1-min data, outputs hourly+daily CSVs and charts
  python main_es.py capacity --days 30     # use 30 days of history for better accuracy
  python main_es.py capacity --forecast-days 60  # forecast 60 days instead of 30
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from config.settings import (
    BATCH_SIZE_ES,
    BATCH_SIZE_ES_HOURLY,
    BATCH_SIZE_ES_DAILY,
    EPOCHS_ES,
    EPOCHS_ES_DAILY,
    EPOCHS_ES_HOURLY,
    ES_CPU_CORES,
    ES_ENTITY_ID,
    ES_FORECASTS_DIR,
    ES_HOST,
    ES_INDEX,
    ES_LOOKBACK_DAYS,
    ES_METRIC_TABLE,
    ES_MODELS_DIR,
    ES_PASSWORD,
    ES_USER,
    FREQ_ES,
    FREQ_ES_DAILY,
    FREQ_ES_HOURLY,
    HOLDOUT_DAYS_ES,
    HOLDOUT_HOURS_ES,
    HOLDOUT_MINUTES_ES,
    N_FORECASTS_ES,
    N_FORECASTS_ES_DAILY,
    N_FORECASTS_ES_HOURLY,
    N_LAGS_ES,
    N_LAGS_ES_DAILY,
    N_LAGS_ES_HOURLY,
)
from pipeline import (
    ESMetricsForecaster,
    describe_es_dataset,
    export_es_dataset,
    load_from_elasticsearch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution config resolver
# ---------------------------------------------------------------------------

_RESOLUTION_CONFIGS = {
    "1min": {
        "freq": FREQ_ES,
        "n_lags": N_LAGS_ES,
        "n_forecasts": N_FORECASTS_ES,
        "epochs": EPOCHS_ES,
        "batch_size": BATCH_SIZE_ES,
        "holdout": HOLDOUT_MINUTES_ES,
        "holdout_label": "minutes",
    },
    "1h": {
        "freq": FREQ_ES_HOURLY,
        "n_lags": N_LAGS_ES_HOURLY,
        "n_forecasts": N_FORECASTS_ES_HOURLY,
        "epochs": EPOCHS_ES_HOURLY,
        "batch_size": BATCH_SIZE_ES_HOURLY,
        "holdout": HOLDOUT_HOURS_ES,
        "holdout_label": "hours",
    },
    "D": {
        "freq": FREQ_ES_DAILY,
        "n_lags": N_LAGS_ES_DAILY,
        "n_forecasts": N_FORECASTS_ES_DAILY,
        "epochs": EPOCHS_ES_DAILY,
        "batch_size": BATCH_SIZE_ES_DAILY,
        "holdout": HOLDOUT_DAYS_ES,
        "holdout_label": "days",
    },
}


def _resolution_cfg(args: argparse.Namespace) -> dict:
    """Return the appropriate hyperparameter config dict for the chosen resolution."""
    res = getattr(args, "resolution", "1min")
    cfg = _RESOLUTION_CONFIGS.get(res)
    if cfg is None:
        raise ValueError(
            f"Unknown --resolution '{res}'. Choose from: 1min, 1h, D"
        )
    # Allow CLI overrides for n-lags, n-forecasts, epochs if explicitly set
    if hasattr(args, "n_lags") and args.n_lags is not None:
        cfg = dict(cfg)
        cfg["n_lags"] = args.n_lags
    if hasattr(args, "n_forecasts") and args.n_forecasts is not None:
        cfg = dict(cfg)
        cfg["n_forecasts"] = args.n_forecasts
    if hasattr(args, "epochs") and args.epochs is not None:
        cfg = dict(cfg)
        cfg["epochs"] = args.epochs
    return cfg


def _get_model_path(entity_id: str, resolution: str = "1min") -> Path:
    safe_entity = re.sub(r"[^\w\-]", "_", entity_id)
    safe_res = resolution.replace("/", "").replace("\\", "")
    return ES_MODELS_DIR / f"model_es_cpu_{safe_entity}_{safe_res}.pt"


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect ES dataset, display summary metrics, and save CSV/Excel exports."""
    logger.info("Running ES dataset inspection...")
    resolution = getattr(args, "resolution", "1min")
    df = load_from_elasticsearch(
        es_host=args.host,
        es_user=args.user,
        es_password=args.password,
        es_index=args.index,
        entity_id=args.entity,
        metric_table=args.metric_table,
        lookback_days=args.days,
        resolution=resolution,
        cpu_cores=getattr(args, "cores", ES_CPU_CORES),
    )

    describe_es_dataset(df, entity_id=args.entity)

    if args.export:
        csv_path, xlsx_path = export_es_dataset(df, entity_id=args.entity)
        print(f"Exported CSV file:   {csv_path}")
        print(f"Exported Excel file: {xlsx_path}\n")


def cmd_fit(args: argparse.Namespace) -> None:
    """Train NeuralProphet model on ES metric dataset at the chosen resolution."""
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        from main_hw import cmd_es_fit
        cmd_es_fit(args)
        return

    resolution = getattr(args, "resolution", "1min")
    cfg = _resolution_cfg(args)
    logger.info("Running ES model fitting at resolution='%s'...", resolution)
    df = load_from_elasticsearch(
        es_host=args.host,
        es_user=args.user,
        es_password=args.password,
        es_index=args.index,
        entity_id=args.entity,
        metric_table=args.metric_table,
        lookback_days=args.days,
        resolution=resolution,
        cpu_cores=getattr(args, "cores", ES_CPU_CORES),
    )

    print(f"\nResolution '{resolution}' — {len(df):,} rows, "
          f"n_lags={cfg['n_lags']}, n_forecasts={cfg['n_forecasts']}, epochs={cfg['epochs']}")

    forecaster = ESMetricsForecaster(
        n_lags=cfg["n_lags"],
        n_forecasts=cfg["n_forecasts"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        freq=cfg["freq"],
    )
    forecaster.fit(df)

    model_path = _get_model_path(args.entity, resolution)
    forecaster.save(model_path)
    print(f"\nModel fitting complete! Model saved to: {model_path}\n")


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate future predictions and save forecast charts."""
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        from main_hw import cmd_es_predict
        cmd_es_predict(args)
        return

    resolution = getattr(args, "resolution", "1min")
    cfg = _resolution_cfg(args)
    logger.info("Running ES forecast prediction at resolution='%s'...", resolution)
    df = load_from_elasticsearch(
        es_host=args.host,
        es_user=args.user,
        es_password=args.password,
        es_index=args.index,
        entity_id=args.entity,
        metric_table=args.metric_table,
        lookback_days=args.days,
        resolution=resolution,
        cpu_cores=getattr(args, "cores", ES_CPU_CORES),
    )

    model_path = _get_model_path(args.entity, resolution)
    forecaster = ESMetricsForecaster(
        n_lags=cfg["n_lags"],
        n_forecasts=cfg["n_forecasts"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        freq=cfg["freq"],
    )

    if model_path.exists():
        forecaster.load(model_path)
    else:
        logger.info("No saved model found at %s. Fitting a new model...", model_path)
        forecaster.fit(df)
        forecaster.save(model_path)

    forecast_df = forecaster.predict(df, periods=cfg["n_forecasts"])

    # Export forecast CSV
    safe_entity = re.sub(r"[^\w\-]", "_", args.entity)
    safe_res = resolution.replace("/", "")
    forecast_csv = ES_FORECASTS_DIR / f"forecast_es_{safe_entity}_{safe_res}.csv"
    ES_FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_csv, index=False)

    # Plot forecast image
    plot_file = forecaster.plot_forecast(
        df, forecast_df, entity_id=args.entity,
        output_file=ES_FORECASTS_DIR / f"forecast_{safe_entity}_{safe_res}.png",
    )

    horizon_label = {
        "1min": f"{cfg['n_forecasts']} minutes",
        "1h": f"{cfg['n_forecasts'] // 24} days ({cfg['n_forecasts']} hours)",
        "D": f"{cfg['n_forecasts']} days",
    }.get(resolution, f"{cfg['n_forecasts']} steps")

    print("\n" + "=" * 60)
    print(f" FORECAST GENERATED — Resolution: {resolution} | Horizon: {horizon_label}")
    print(f" Entity: {args.entity}")
    print("=" * 60)
    print(f" Forecast CSV  : {forecast_csv}")
    print(f" Forecast Plot : {plot_file}")
    print("=" * 60 + "\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model performance on a trailing holdout period."""
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        from main_hw import cmd_es_evaluate
        cmd_es_evaluate(args)
        return

    resolution = getattr(args, "resolution", "1min")
    cfg = _resolution_cfg(args)
    logger.info("Running ES model evaluation at resolution='%s'...", resolution)
    df = load_from_elasticsearch(
        es_host=args.host,
        es_user=args.user,
        es_password=args.password,
        es_index=args.index,
        entity_id=args.entity,
        metric_table=args.metric_table,
        lookback_days=args.days,
        resolution=resolution,
        cpu_cores=getattr(args, "cores", ES_CPU_CORES),
    )

    forecaster = ESMetricsForecaster(
        n_lags=cfg["n_lags"],
        n_forecasts=cfg["n_forecasts"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        freq=cfg["freq"],
    )

    holdout = getattr(args, "holdout", None) or cfg["holdout"]
    results = forecaster.evaluate(df, holdout_minutes=holdout)
    print(f"Evaluation completed! Holdout: {holdout} {cfg['holdout_label']}")


def cmd_capacity(args: argparse.Namespace) -> None:
    """
    30-day capacity planning forecast.
    Model is trained on 1-minute ES data.
    Output: hourly + daily aggregated CSV/Excel files and charts.
    """
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        from main_hw import cmd_es_capacity
        cmd_es_capacity(args)
        return

    forecast_days: int = getattr(args, "forecast_days", 30)
    logger.info("Running 30-day capacity planning forecast (1-min model, %d days)...", forecast_days)

    # Always load 1-minute resolution data for training
    df = load_from_elasticsearch(
        es_host=args.host,
        es_user=args.user,
        es_password=args.password,
        es_index=args.index,
        entity_id=args.entity,
        metric_table=args.metric_table,
        lookback_days=args.days,
        resolution="1min",
        cpu_cores=getattr(args, "cores", ES_CPU_CORES),
    )
    logger.info("Loaded %d 1-minute rows for training.", len(df))

    model_path = _get_model_path(args.entity, "1min")
    forecaster = ESMetricsForecaster(
        n_lags=N_LAGS_ES,
        n_forecasts=N_FORECASTS_ES,
        epochs=EPOCHS_ES,
        batch_size=BATCH_SIZE_ES,
        freq=FREQ_ES,
    )

    if model_path.exists():
        logger.info("Loading existing 1-min model from: %s", model_path)
        forecaster.load(model_path)
    else:
        logger.info("No saved 1-min model found. Fitting on %d rows...", len(df))
        forecaster.fit(df)
        forecaster.save(model_path)

    # Generate 30-day capacity forecast (returns minutely, hourly, daily DataFrames)
    capacity = forecaster.predict_capacity(df, forecast_days=forecast_days)

    fut_hourly = capacity["hourly"]
    fut_daily  = capacity["daily"]
    fut_1min   = capacity["minutely"]

    # --- Export outputs ---
    safe_entity = re.sub(r"[^\w\-]", "_", args.entity)
    cap_dir = ES_FORECASTS_DIR / "capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)

    # Hourly CSV + Excel
    hourly_csv  = cap_dir / f"capacity_hourly_{safe_entity}.csv"
    hourly_xlsx = cap_dir / f"capacity_hourly_{safe_entity}.xlsx"
    fut_hourly.to_csv(hourly_csv, index=False)
    fut_hourly.to_excel(hourly_xlsx, index=False)

    # Daily CSV + Excel
    daily_csv  = cap_dir / f"capacity_daily_{safe_entity}.csv"
    daily_xlsx = cap_dir / f"capacity_daily_{safe_entity}.xlsx"
    fut_daily.to_csv(daily_csv, index=False)
    fut_daily.to_excel(daily_xlsx, index=False)

    # Minute-level CSV (only present when using rolling 1-min approach)
    minutely_csv = cap_dir / f"capacity_minutely_{safe_entity}.csv"
    if not fut_1min.empty:
        fut_1min.to_csv(minutely_csv, index=False)

    # Generate and save charts
    chart_paths = forecaster.plot_capacity_forecast(
        df=df,
        capacity_forecasts=capacity,
        entity_id=args.entity,
        output_dir=cap_dir,
        forecast_days=forecast_days,
    )

    print("\n" + "=" * 65)
    print(f" 30-DAY CAPACITY FORECAST — Model: 1-min | Horizon: {forecast_days} days")
    print(f" Entity: {args.entity}")
    print("=" * 65)
    print(f" Hourly CSV   : {hourly_csv}")
    print(f" Hourly Excel : {hourly_xlsx}")
    print(f" Daily  CSV   : {daily_csv}")
    print(f" Daily  Excel : {daily_xlsx}")
    print(f" 1-min  CSV   : {minutely_csv}")
    if "hourly" in chart_paths:
        print(f" Hourly Chart : {chart_paths['hourly']}")
    if "daily" in chart_paths:
        print(f" Daily  Chart : {chart_paths['daily']}")
    print("=" * 65)
    print()
    if not fut_daily.empty:
        print(" DAILY SUMMARY TABLE:")
        print(" {:<12} {:>12} {:>12} {:>12}".format("Date", "Mean CPU%", "Peak CPU%", "Min CPU%"))
        print(" " + "-" * 52)
        for _, row in fut_daily.iterrows():
            print(" {:<12} {:>12.1f} {:>12.1f} {:>12.1f}".format(
                str(row["ds"])[:10], row["yhat_mean"], row["yhat_max"], row["yhat_min"]
            ))
    print()


def main() -> None:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--host", type=str, default=ES_HOST, help="Elasticsearch server base URL")
    parent_parser.add_argument("--user", type=str, default=ES_USER, help="Elasticsearch username")
    parent_parser.add_argument("--password", type=str, default=ES_PASSWORD, help="Elasticsearch password")
    parent_parser.add_argument("--index", type=str, default=ES_INDEX, help="Elasticsearch index pattern")
    parent_parser.add_argument("--entity", type=str, default=ES_ENTITY_ID, help="Target entity_id")
    parent_parser.add_argument(
        "--metric-table", type=str, default=ES_METRIC_TABLE, help="Target metric table name"
    )
    parent_parser.add_argument("--days", type=int, default=ES_LOOKBACK_DAYS, help="Lookback days window")
    parent_parser.add_argument(
        "--cores",
        type=float,
        default=ES_CPU_CORES,
        help="Number of CPU cores to divide multi-core percentage metric by (default: 6)",
    )
    parent_parser.add_argument(
        "--resolution",
        type=str,
        default="1min",
        choices=["1min", "1h", "D"],
        help="Time resolution: '1min' (default, 60-step forecast), '1h' (hourly, 30-day forecast), 'D' (daily, 30-day forecast)",
    )
    # Optional overrides (default None so resolution config is used)
    parent_parser.add_argument("--n-lags", type=int, default=None, help="Override AR lookback lags")
    parent_parser.add_argument("--n-forecasts", type=int, default=None, help="Override forecast horizon steps")
    parent_parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parent_parser.add_argument(
        "--model",
        type=str,
        default="neuralprophet",
        choices=["neuralprophet", "holt-winters", "hw"],
        help="Forecasting model to use: 'neuralprophet' (default) or 'holt-winters'",
    )
    parent_parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional local CSV file path to load data from instead of live ES",
    )

    parser = argparse.ArgumentParser(
        prog="main_es.py",
        description="Elasticsearch Infrastructure Metrics Forecasting Pipeline (1min / Hourly / Daily)",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=[parent_parser],
        help="Inspect ES metric data and summary stats",
    )
    inspect_parser.add_argument(
        "--export", action="store_true", default=True, help="Export dataset to CSV and Excel files"
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # fit command
    subparsers.add_parser(
        "fit",
        parents=[parent_parser],
        help="Train NeuralProphet model on ES metrics",
    ).set_defaults(func=cmd_fit)

    # predict command
    subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Generate N-step future forecasts and plot chart",
    ).set_defaults(func=cmd_predict)

    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[parent_parser],
        help="Evaluate model accuracy on holdout test data",
    )
    eval_parser.add_argument(
        "--holdout", type=int, default=None,
        help="Holdout window size (units depend on resolution: minutes / hours / days)",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # capacity command (30-day planning forecast: 1-min model → hourly + daily output)
    cap_parser = subparsers.add_parser(
        "capacity",
        parents=[parent_parser],
        help=(
            "30-day capacity planning forecast — trains on 1-minute ES data, "
            "outputs hourly and daily aggregated CSV/Excel + charts"
        ),
    )
    cap_parser.add_argument(
        "--forecast-days", type=int, default=30,
        help="Number of future days to forecast (default: 30)",
    )
    cap_parser.set_defaults(func=cmd_capacity)

    # compare-models command (NeuralProphet vs Holt-Winters side-by-side)
    def _cmd_compare(args: argparse.Namespace) -> None:
        from main_hw import cmd_es_compare
        cmd_es_compare(args)

    cmp_parser = subparsers.add_parser(
        "compare-models",
        parents=[parent_parser],
        help="Compare NeuralProphet vs Holt-Winters side-by-side on holdout data",
    )
    cmp_parser.add_argument(
        "--holdout", type=int, default=None,
        help="Holdout window size (minutes / hours / days depending on resolution)",
    )
    cmp_parser.set_defaults(func=_cmd_compare)

    args = parser.parse_args()
    args.func(args)



if __name__ == "__main__":
    main()
