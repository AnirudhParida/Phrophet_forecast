"""
cli/cli_holtwinters.py
======================
Command-line interface for Holt-Winters Exponential Smoothing across CSV and Elasticsearch pipelines.
"""

from __future__ import annotations

import argparse
import logging
import re
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
    FORECASTS_HW_CSV_DIR,
    FORECASTS_HW_ES_DIR,
    HOLDOUT_DAYS,
    HOSTS,
    HW_CSV_MODELS_DIR,
    HW_ES_MODELS_DIR,
    METRICS,
    N_FORECASTS,
)
from pipeline.ingestion import load_and_preprocess, load_from_elasticsearch
from pipeline.models.holt_winters import (
    ESHoltWintersForecaster,
    HoltWintersForecaster,
)

logger = logging.getLogger("cli_hw")


# ---------------------------------------------------------------------------
# CSV Pipeline Handlers
# ---------------------------------------------------------------------------

def cmd_hw_csv_fit(args: argparse.Namespace) -> None:
    """Train Holt-Winters models for CSV/Excel server host metrics."""
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]

    for host in target_hosts:
        if host not in datasets:
            continue
        df = datasets[host]
        logger.info("\nFitting Holt-Winters for host: %s (%d rows)...", host, len(df))
        forecaster = HoltWintersForecaster()
        forecaster.fit(df)
        saved_paths = forecaster.save(HW_CSV_MODELS_DIR, host_alias=host)
        print(f"\n✅ Successfully fitted Holt-Winters for {host} ({len(saved_paths)} metric models saved to {HW_CSV_MODELS_DIR})")


def cmd_hw_csv_predict(args: argparse.Namespace) -> None:
    """Generate future forecast CSVs and PNG charts using Holt-Winters."""
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    periods = getattr(args, "periods", N_FORECASTS)

    for host in target_hosts:
        if host not in datasets:
            continue
        df = datasets[host]
        forecaster = HoltWintersForecaster(n_forecasts=periods)
        forecaster.load(HW_CSV_MODELS_DIR, host_alias=host)

        forecasts = forecaster.predict(df, periods=periods)
        if not forecasts:
            logger.warning("No forecasts generated for host '%s'", host)
            continue

        # Save any models that were fitted on the fly
        forecaster.save(HW_CSV_MODELS_DIR, host_alias=host)
        FORECASTS_HW_CSV_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}\n  HOLT-WINTERS FORECAST: {host} ({periods} days ahead)\n{'='*65}")

        combined_fc = pd.DataFrame({"ds": forecasts[list(forecasts.keys())[0]]["ds"]})
        for m, fc in forecasts.items():
            combined_fc[f"{m}_yhat"] = fc["yhat"]
            combined_fc[f"{m}_lower"] = fc["yhat_lower"]
            combined_fc[f"{m}_upper"] = fc["yhat_upper"]

            plot_path = forecaster.plot_forecast(host, m, df, fc, output_file=FORECASTS_HW_CSV_DIR / f"forecast_hw_{host}_{m}.png")
            print(f"  Chart ({m:<18}): {plot_path}")

        csv_path = FORECASTS_HW_CSV_DIR / f"forecast_hw_{host}_{periods}d.csv"
        combined_fc.to_csv(csv_path, index=False)
        print(f"\n  Combined Forecast CSV: {csv_path}\n{'='*65}\n")


def cmd_hw_csv_evaluate(args: argparse.Namespace) -> None:
    """Evaluate Holt-Winters forecast accuracy on a trailing holdout window."""
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    holdout = getattr(args, "holdout", HOLDOUT_DAYS)

    for host in target_hosts:
        if host not in datasets:
            continue
        df = datasets[host]
        forecaster = HoltWintersForecaster()
        results = forecaster.evaluate(df, holdout_days=holdout)

        print(f"\n{'='*75}\n  Holt-Winters Evaluation: {host}  |  Holdout: {holdout} days\n{'='*75}")
        print(f"  {'Metric':<18} {'MAE':>14} {'RMSE':>14} {'WAPE':>10} {'MAPE':>10}")
        print(f"  {'-'*68}")
        for metric, scores in results.items():
            if metric.endswith("_bytes"):
                mae_str = f"{scores['MAE'] / 1024.0:.2f} KB/s"
                rmse_str = f"{scores['RMSE'] / 1024.0:.2f} KB/s"
            else:
                mae_str = f"{scores['MAE']:.2f} pp"
                rmse_str = f"{scores['RMSE']:.2f} pp"
            wape_str = f"{scores.get('WAPE', 0.0):.2f}%"
            mape_str = f"{scores['MAPE']:.2f}%"
            print(f"  {metric:<18} {mae_str:>14} {rmse_str:>14} {wape_str:>10} {mape_str:>10}")
        print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# ES Pipeline Handlers
# ---------------------------------------------------------------------------

def _load_es_df(args: argparse.Namespace) -> pd.DataFrame:
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
        return df

    return load_from_elasticsearch(
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


def cmd_hw_es_fit(args: argparse.Namespace) -> None:
    """Fit Holt-Winters on ES metrics."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)

    forecaster = ESHoltWintersForecaster(resolution=resolution)
    forecaster.fit(df)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    model_path = HW_ES_MODELS_DIR / f"hw_es_{safe_entity}_{resolution}.joblib"
    forecaster.save(model_path)
    print(f"\n✅ Successfully fitted ES Holt-Winters model ({resolution}). Saved to: {model_path}\n")


def cmd_hw_es_predict(args: argparse.Namespace) -> None:
    """Generate future forecast CSV and plot chart for ES metrics."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)

    periods = getattr(args, "periods", None)
    forecaster = ESHoltWintersForecaster(resolution=resolution, n_forecasts=periods)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    model_path = HW_ES_MODELS_DIR / f"hw_es_{safe_entity}_{resolution}.joblib"

    if model_path.exists():
        forecaster.load(model_path)
    else:
        logger.info("No saved model found at %s. Fitting on current data...", model_path)
        forecaster.fit(df)
        forecaster.save(model_path)

    forecast_df = forecaster.predict(df, periods=periods)

    FORECASTS_HW_ES_DIR.mkdir(parents=True, exist_ok=True)
    fc_csv = FORECASTS_HW_ES_DIR / f"forecast_hw_es_{safe_entity}_{resolution}.csv"
    forecast_df.to_csv(fc_csv, index=False)

    plot_path = forecaster.plot_forecast(
        df, forecast_df, entity_id=getattr(args, "entity", ES_ENTITY_ID),
        output_file=FORECASTS_HW_ES_DIR / f"forecast_hw_es_{safe_entity}_{resolution}.png"
    )

    print(f"\n{'='*65}\n  HOLT-WINTERS ES FORECAST ({resolution})\n{'='*65}")
    print(f"  Forecast CSV  : {fc_csv}")
    print(f"  Forecast Plot : {plot_path}")
    print(f"{'='*65}\n")


def cmd_hw_es_evaluate(args: argparse.Namespace) -> None:
    """Evaluate Holt-Winters ES model on holdout."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)
    holdout = getattr(args, "holdout", None)

    forecaster = ESHoltWintersForecaster(resolution=resolution)
    forecaster.evaluate(df, holdout_steps=holdout)


def cmd_hw_es_capacity(args: argparse.Namespace) -> None:
    """Run 30-day capacity forecast with Holt-Winters."""
    args.resolution = "1min"
    df = _load_es_df(args)

    forecaster = ESHoltWintersForecaster(resolution="1min")
    forecaster.fit(df)

    forecast_days = getattr(args, "forecast_days", 30)
    capacity = forecaster.predict_capacity(df, forecast_days=forecast_days)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    cap_dir = FORECASTS_HW_ES_DIR / "capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)

    hourly_csv = cap_dir / f"capacity_hw_hourly_{safe_entity}.csv"
    daily_csv = cap_dir / f"capacity_hw_daily_{safe_entity}.csv"
    hourly_xlsx = cap_dir / f"capacity_hw_hourly_{safe_entity}.xlsx"
    daily_xlsx = cap_dir / f"capacity_hw_daily_{safe_entity}.xlsx"

    capacity["hourly"].to_csv(hourly_csv, index=False)
    capacity["hourly"].to_excel(hourly_xlsx, index=False)
    capacity["daily"].to_csv(daily_csv, index=False)
    capacity["daily"].to_excel(daily_xlsx, index=False)

    print(f"\n{'='*65}\n  HOLT-WINTERS 30-DAY CAPACITY FORECAST COMPLETE\n{'='*65}")
    print(f"  Hourly CSV   : {hourly_csv}")
    print(f"  Hourly Excel : {hourly_xlsx}")
    print(f"  Daily CSV    : {daily_csv}")
    print(f"  Daily Excel  : {daily_xlsx}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Parser Setup
# ---------------------------------------------------------------------------

def build_hw_parser(subparsers=None) -> argparse.ArgumentParser:
    """Build Holt-Winters CLI parser."""
    if subparsers is not None:
        p = subparsers.add_parser("hw", help="Holt-Winters forecasting pipeline")
    else:
        p = argparse.ArgumentParser(prog="main_hw.py", description="Holt-Winters CLI (CSV & Elasticsearch)")

    pipe_subs = p.add_subparsers(dest="pipeline", required=True)

    # --- CSV Subcommands ---
    csv_p = pipe_subs.add_parser("csv", help="Holt-Winters on CSV/Excel host metrics")
    csv_subs = csv_p.add_subparsers(dest="command", required=True)

    p_fit = csv_subs.add_parser("fit", help="Fit Holt-Winters models")
    p_fit.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_fit.add_argument("--all", action="store_true", help="Fit all hosts")
    p_fit.set_defaults(func=cmd_hw_csv_fit)

    p_pred = csv_subs.add_parser("predict", help="Generate forecast charts")
    p_pred.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_pred.add_argument("--all", action="store_true", help="Predict all hosts")
    p_pred.add_argument("--periods", type=int, default=N_FORECASTS, help="Forecast horizon (days)")
    p_pred.set_defaults(func=cmd_hw_csv_predict)

    p_eval = csv_subs.add_parser("evaluate", help="Evaluate holdout error metrics")
    p_eval.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_eval.add_argument("--all", action="store_true", help="Evaluate all hosts")
    p_eval.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout days")
    p_eval.set_defaults(func=cmd_hw_csv_evaluate)

    # --- ES Subcommands ---
    es_p = pipe_subs.add_parser("es", help="Holt-Winters on Elasticsearch metrics")
    es_subs = es_p.add_subparsers(dest="command", required=True)

    for p_name, func in [
        ("fit", cmd_hw_es_fit),
        ("predict", cmd_hw_es_predict),
        ("evaluate", cmd_hw_es_evaluate),
        ("capacity", cmd_hw_es_capacity),
    ]:
        sp = es_subs.add_parser(p_name, help=f"{p_name.capitalize()} on ES data")
        sp.add_argument("--resolution", default="1min", choices=["1min", "1h", "D"], help="Data resolution")
        sp.add_argument("--csv-path", default=None, help="Optional CSV file path")
        sp.add_argument("--days", type=int, default=ES_LOOKBACK_DAYS, help="Lookback days for live ES")
        sp.add_argument("--entity", default=ES_ENTITY_ID, help="Entity ID")
        sp.add_argument("--host", default=ES_HOST, help="ES Host URL")
        sp.add_argument("--user", default=ES_USER, help="ES User")
        sp.add_argument("--password", default=ES_PASSWORD, help="ES Password")
        sp.add_argument("--index", default=ES_INDEX, help="ES Index")
        sp.add_argument("--metric-table", default=ES_METRIC_TABLE, help="Metric table")
        sp.add_argument("--cores", type=float, default=ES_CPU_CORES, help="Host CPU core count")
        if p_name in ["predict", "evaluate"]:
            sp.add_argument("--periods", type=int, default=None, help="Steps")
        if p_name in ["evaluate"]:
            sp.add_argument("--holdout", type=int, default=None, help="Holdout steps")
        if p_name == "capacity":
            sp.add_argument("--forecast-days", type=int, default=30, help="Days to forecast")
        sp.set_defaults(func=func)

    return p


def main() -> None:
    parser = build_hw_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
