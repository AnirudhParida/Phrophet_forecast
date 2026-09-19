"""
cli/cli_neuralprophet.py
========================
Command-line interface for NeuralProphet forecasting across CSV and Elasticsearch pipelines.
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
    FORECASTS_NP_CSV_DIR,
    FORECASTS_NP_ES_DIR,
    HOLDOUT_DAYS,
    HOSTS,
    NP_CSV_MODELS_DIR,
    NP_ES_MODELS_DIR,
    N_FORECASTS,
    N_FORECASTS_ES,
    N_LAGS_ES,
)
from pipeline.ingestion import load_and_preprocess, load_from_elasticsearch
from pipeline.models.neuralprophet import (
    ESMetricsForecaster,
    ServerMetricsForecaster,
    extract_es_forecast_rows,
)
from pipeline.common.visualization import extract_forecast_rows

logger = logging.getLogger("cli_np")


# ---------------------------------------------------------------------------
# CSV Pipeline Handlers
# ---------------------------------------------------------------------------

def cmd_np_csv_fit(args: argparse.Namespace) -> None:
    """Train NeuralProphet models for one or all hosts."""
    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=getattr(args, "n_forecasts", N_FORECASTS))
    forecaster.load_and_preprocess()

    if getattr(args, "parallel", False) or getattr(args, "all", False):
        host_str = "ALL HOSTS"
    else:
        if not args.host:
            print("❌ Specify --host <HOST> or use --all. Exiting.")
            sys.exit(1)
        host_str = args.host

    print(f"\n🚀 Fitting NeuralProphet models: {host_str} | n_lags={args.n_lags or 'auto'}")
    if getattr(args, "parallel", False):
        forecaster.fit(host_name=None, n_lags=args.n_lags, parallel=True)
    elif getattr(args, "all", False):
        for alias in HOSTS:
            forecaster.fit(host_name=alias, n_lags=args.n_lags)
    else:
        forecaster.fit(host_name=args.host, n_lags=args.n_lags)

    print(f"\n✅ Training complete. Models saved to {NP_CSV_MODELS_DIR}/\n")


def cmd_np_csv_predict(args: argparse.Namespace) -> None:
    """Generate NeuralProphet forecasts and charts."""
    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=getattr(args, "periods", N_FORECASTS))
    forecaster.load_and_preprocess()

    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]

    for host in target_hosts:
        print(f"\n📈 Predicting with NeuralProphet: {host} | n_lags={args.n_lags or 'auto'}")
        forecasts = forecaster.predict(host_name=host, n_lags=args.n_lags, save_charts=True)

        print(f"\n📋 Forecast Table for {host}:\n")
        for metric, f_df in forecasts.items():
            clean_df = extract_forecast_rows(f_df)
            if clean_df is not None:
                print(f"--- {host} | {metric} ---")
                for _, row in clean_df.head(5).iterrows():
                    ds_str = pd.to_datetime(row["ds"]).strftime("%Y-%m-%d")
                    yhat_str = f"{row['yhat']:6.2f}%"
                    print(f"  {ds_str}  |  forecast: {yhat_str}")
                print(f"  ... ({len(clean_df)} total forecast days)\n")


def cmd_np_csv_evaluate(args: argparse.Namespace) -> None:
    """Run holdout evaluation for NeuralProphet on host metrics."""
    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    holdout = getattr(args, "holdout", HOLDOUT_DAYS)

    for host in target_hosts:
        print(f"\n📊 Evaluating NeuralProphet: {host} | holdout={holdout} days")
        forecaster.evaluate(host_name=host, holdout_days=holdout, n_lags=args.n_lags)


def cmd_np_csv_compare_lags(args: argparse.Namespace) -> None:
    """Compare n_lags=7 vs n_lags=14 on holdout data."""
    forecaster = ServerMetricsForecaster(n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    print(f"\n🔬 Comparing n_lags=7 vs n_lags=14 for: {args.host}")
    forecaster.compare_lags(host_name=args.host, holdout_days=getattr(args, "holdout", HOLDOUT_DAYS))


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


def cmd_np_es_fit(args: argparse.Namespace) -> None:
    """Train NeuralProphet on ES metrics."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)

    forecaster = ESMetricsForecaster(
        n_lags=getattr(args, "n_lags", None) or (N_LAGS_ES if resolution == "1min" else 24),
        n_forecasts=getattr(args, "n_forecasts", None) or (N_FORECASTS_ES if resolution == "1min" else 24),
        epochs=getattr(args, "epochs", None) or 50,
        freq="1min" if resolution == "1min" else ("1h" if resolution == "1h" else "D"),
    )
    forecaster.fit(df)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    model_path = NP_ES_MODELS_DIR / f"np_es_{safe_entity}_{resolution}.pt"
    forecaster.save(model_path)
    print(f"\n✅ NeuralProphet ES model saved to: {model_path}\n")


def cmd_np_es_predict(args: argparse.Namespace) -> None:
    """Generate forecast with NeuralProphet on ES metrics."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    model_path = NP_ES_MODELS_DIR / f"np_es_{safe_entity}_{resolution}.pt"

    forecaster = ESMetricsForecaster(
        freq="1min" if resolution == "1min" else ("1h" if resolution == "1h" else "D")
    )
    if model_path.exists():
        forecaster.load(model_path)
    else:
        logger.info("Fitting new NeuralProphet model on current data...")
        forecaster.fit(df)
        forecaster.save(model_path)

    periods = getattr(args, "periods", None) or (N_FORECASTS_ES if resolution == "1min" else 24)
    forecast_df = forecaster.predict(df, periods=periods)

    FORECASTS_NP_ES_DIR.mkdir(parents=True, exist_ok=True)
    fc_csv = FORECASTS_NP_ES_DIR / f"forecast_np_es_{safe_entity}_{resolution}.csv"
    forecast_df.to_csv(fc_csv, index=False)

    plot_path = forecaster.plot_forecast(
        df, forecast_df, entity_id=getattr(args, "entity", ES_ENTITY_ID),
        output_file=FORECASTS_NP_ES_DIR / f"forecast_np_es_{safe_entity}_{resolution}.png"
    )

    print(f"\n{'='*65}\n  NEURALPROPHET ES FORECAST ({resolution})\n{'='*65}")
    print(f"  Forecast CSV  : {fc_csv}")
    print(f"  Forecast Plot : {plot_path}")
    print(f"{'='*65}\n")


def cmd_np_es_evaluate(args: argparse.Namespace) -> None:
    """Evaluate NeuralProphet accuracy on ES holdout."""
    resolution = getattr(args, "resolution", "1min")
    df = _load_es_df(args)

    forecaster = ESMetricsForecaster(
        freq="1min" if resolution == "1min" else ("1h" if resolution == "1h" else "D")
    )
    holdout = getattr(args, "holdout", None) or (120 if resolution == "1min" else 24)
    forecaster.evaluate(df, holdout_minutes=holdout)


def cmd_np_es_capacity(args: argparse.Namespace) -> None:
    """30-day capacity forecast with NeuralProphet."""
    args.resolution = "1min"
    df = _load_es_df(args)

    forecaster = ESMetricsForecaster(freq="1min")
    forecaster.fit(df)

    forecast_days = getattr(args, "forecast_days", 30)
    capacity = forecaster.predict_capacity(df, forecast_days=forecast_days)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    cap_dir = FORECASTS_NP_ES_DIR / "capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)

    hourly_csv = cap_dir / f"capacity_np_hourly_{safe_entity}.csv"
    daily_csv = cap_dir / f"capacity_np_daily_{safe_entity}.csv"
    capacity["hourly"].to_csv(hourly_csv, index=False)
    capacity["daily"].to_csv(daily_csv, index=False)

    print(f"\n✅ 30-Day Capacity Forecast saved to:\n  Hourly: {hourly_csv}\n  Daily : {daily_csv}\n")


# ---------------------------------------------------------------------------
# Parser Setup
# ---------------------------------------------------------------------------

def build_np_parser(subparsers=None) -> argparse.ArgumentParser:
    """Build NeuralProphet CLI parser."""
    if subparsers is not None:
        p = subparsers.add_parser("np", help="NeuralProphet forecasting pipeline")
    else:
        p = argparse.ArgumentParser(prog="main_np.py", description="NeuralProphet CLI (CSV & Elasticsearch)")

    pipe_subs = p.add_subparsers(dest="pipeline", required=True)

    # --- CSV Subcommands ---
    csv_p = pipe_subs.add_parser("csv", help="NeuralProphet on CSV/Excel host metrics")
    csv_subs = csv_p.add_subparsers(dest="command", required=True)

    p_fit = csv_subs.add_parser("fit", help="Train NeuralProphet models")
    p_fit.add_argument("--host", choices=list(HOSTS.keys()), help="Target host")
    p_fit.add_argument("--all", action="store_true", help="Fit all hosts")
    p_fit.add_argument("--parallel", action="store_true", help="Fit in parallel")
    p_fit.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_fit.set_defaults(func=cmd_np_csv_fit)

    p_pred = csv_subs.add_parser("predict", help="Generate forecast charts")
    p_pred.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_pred.add_argument("--all", action="store_true", help="Predict all hosts")
    p_pred.add_argument("--periods", type=int, default=N_FORECASTS, help="Forecast horizon (days)")
    p_pred.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_pred.set_defaults(func=cmd_np_csv_predict)

    p_eval = csv_subs.add_parser("evaluate", help="Evaluate holdout error metrics")
    p_eval.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_eval.add_argument("--all", action="store_true", help="Evaluate all hosts")
    p_eval.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout days")
    p_eval.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_eval.set_defaults(func=cmd_np_csv_evaluate)

    p_lag = csv_subs.add_parser("compare-lags", help="Compare n_lags=7 vs 14")
    p_lag.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_lag.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout days")
    p_lag.set_defaults(func=cmd_np_csv_compare_lags)

    # --- ES Subcommands ---
    es_p = pipe_subs.add_parser("es", help="NeuralProphet on Elasticsearch metrics")
    es_subs = es_p.add_subparsers(dest="command", required=True)

    for p_name, func in [
        ("fit", cmd_np_es_fit),
        ("predict", cmd_np_es_predict),
        ("evaluate", cmd_np_es_evaluate),
        ("capacity", cmd_np_es_capacity),
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
        sp.add_argument("--n-lags", type=int, default=None, help="AR lookback lags")
        sp.add_argument("--n-forecasts", type=int, default=None, help="Forecast horizon")
        sp.add_argument("--epochs", type=int, default=None, help="Epochs")
        if p_name in ["predict", "evaluate"]:
            sp.add_argument("--periods", type=int, default=None, help="Steps")
        if p_name in ["evaluate"]:
            sp.add_argument("--holdout", type=int, default=None, help="Holdout steps")
        if p_name == "capacity":
            sp.add_argument("--forecast-days", type=int, default=30, help="Days to forecast")
        sp.set_defaults(func=func)

    return p


def main() -> None:
    parser = build_np_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
