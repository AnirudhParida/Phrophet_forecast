"""
cli/cli_chronos.py
==================
Command-line interface for Amazon Chronos-2 foundation model across CSV and Elasticsearch pipelines.

Supports:
- Universal multi-metric forecasting for server host metrics (CSV/Excel).
- Fine-grained minute, hourly, and daily forecasting for Elasticsearch streaming metrics.
- Multivariate cross-learning via Group Attention mechanism (--multivariate vs --univariate).
- Automatic hardware selection ('auto', 'cpu', 'cuda') ready for local CPU execution and GPU servers.
- Holdout accuracy evaluations (MAE, RMSE, WAPE, MAPE).
- 30-day capacity planning projections and visualization charts.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from config.settings import (
    CHRONOS_DEVICE,
    CHRONOS_HOLDOUT_DAYS_CSV,
    CHRONOS_HORIZON_CSV,
    CHRONOS_MODEL_ID,
    ES_CPU_CORES,
    ES_ENTITY_ID,
    ES_HOST,
    ES_INDEX,
    ES_LOOKBACK_DAYS,
    ES_METRIC_TABLE,
    ES_PASSWORD,
    ES_USER,
    FORECASTS_CHRONOS_CSV_DIR,
    FORECASTS_CHRONOS_ES_DIR,
    HOSTS,
    METRICS,
)
from pipeline.ingestion import load_and_preprocess, load_from_elasticsearch
from pipeline.models.chronos import (
    ChronosCSVForecaster,
    ESChronosForecaster,
    resolve_device,
)

logger = logging.getLogger("cli_chronos")


# ---------------------------------------------------------------------------
# CSV Pipeline Handlers
# ---------------------------------------------------------------------------

def cmd_chronos_csv_fit(args: argparse.Namespace) -> None:
    """Warm up Chronos-2 for CSV host metrics and record baseline metadata."""
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]

    for host in target_hosts:
        if host not in datasets:
            logger.warning("Host '%s' not found in ingested datasets.", host)
            continue
        df = datasets[host]
        logger.info(
            "\nInitializing Chronos-2 for host: %s (%d rows, device=%s, multivariate=%s)...",
            host,
            len(df),
            resolved,
            multivariate,
        )
        forecaster = ChronosCSVForecaster(device=device, multivariate=multivariate)
        forecaster.fit(df, host_alias=host)
        print(
            f"\n✅ Successfully initialized Chronos-2 on {resolved.upper()} for {host} "
            f"(mode: {'Multivariate' if multivariate else 'Univariate'})"
        )


def cmd_chronos_csv_predict(args: argparse.Namespace) -> None:
    """Generate multi-step future forecasts and dark-themed charts with Chronos-2."""
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    periods = getattr(args, "periods", CHRONOS_HORIZON_CSV)

    for host in target_hosts:
        if host not in datasets:
            logger.warning("Host '%s' not found in ingested datasets.", host)
            continue
        df = datasets[host]
        forecaster = ChronosCSVForecaster(
            device=device,
            n_forecasts=periods,
            multivariate=multivariate,
        )
        forecaster.fit(df, host_alias=host)

        forecasts = forecaster.predict(df, periods=periods, host_alias=host)
        if not forecasts:
            logger.warning("No forecasts generated for host '%s'", host)
            continue

        csv_path, xlsx_path = forecaster.save_forecast(forecasts, host_alias=host)
        chart_path = forecaster.plot_forecast(df, forecasts, host_alias=host)

        mode_str = "Multivariate (Group Attention)" if multivariate else "Univariate"
        print(f"\n{'='*75}\n  AMAZON CHRONOS-2 FORECAST: {host} ({periods} days ahead | {mode_str} | Device: {resolved.upper()})\n{'='*75}")
        print(f"  Forecast CSV  : {csv_path}")
        print(f"  Forecast Excel: {xlsx_path}")
        print(f"  Summary Chart : {chart_path}\n{'='*75}\n")


def cmd_chronos_csv_evaluate(args: argparse.Namespace) -> None:
    """Evaluate Chronos-2 forecast accuracy on a trailing holdout window."""
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    datasets = load_and_preprocess()
    target_hosts = list(HOSTS.keys()) if getattr(args, "all", False) else [args.host]
    holdout = getattr(args, "holdout", CHRONOS_HOLDOUT_DAYS_CSV)

    for host in target_hosts:
        if host not in datasets:
            logger.warning("Host '%s' not found in ingested datasets.", host)
            continue
        df = datasets[host]
        forecaster = ChronosCSVForecaster(device=device, multivariate=multivariate)
        results = forecaster.evaluate(df, test_days=holdout, host_alias=host)

        mode_str = "Multivariate (Group Attention)" if multivariate else "Univariate"
        print(f"\n{'='*75}\n  CHRONOS-2 EVALUATION: {host} (Holdout: {holdout} days | {mode_str} | Device: {resolved.upper()})\n{'='*75}")
        print(f"  {'Metric':<22} {'MAE':>10} {'RMSE':>10} {'WAPE (%)':>10} {'MAPE (%)':>10}")
        print(f"  {'-'*66}")
        for m, res in results.items():
            print(f"  {m:<22} {res.get('mae', 0.0):>10.4f} {res.get('rmse', 0.0):>10.4f} {res.get('wape', 0.0)*100:>9.2f}% {res.get('mape', 0.0):>9.2f}%")
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


def cmd_chronos_es_fit(args: argparse.Namespace) -> None:
    """Warm up Chronos-2 on Elasticsearch metrics."""
    resolution = getattr(args, "resolution", "1min")
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    df = _load_es_df(args)

    forecaster = ESChronosForecaster(resolution=resolution, device=device, multivariate=multivariate)
    forecaster.fit(df)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    print(
        f"\n✅ Successfully initialized Chronos-2 on {resolved.upper()} for ES ({resolution}) "
        f"Entity: {safe_entity} (mode: {'Multivariate' if multivariate else 'Univariate'})\n"
    )


def cmd_chronos_es_predict(args: argparse.Namespace) -> None:
    """Generate future forecast CSV and chart for ES metrics using Chronos-2."""
    resolution = getattr(args, "resolution", "1min")
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    df = _load_es_df(args)

    periods = getattr(args, "periods", None)
    forecaster = ESChronosForecaster(
        resolution=resolution,
        n_forecasts=periods,
        device=device,
        multivariate=multivariate,
    )
    forecaster.fit(df)
    forecast_df = forecaster.predict(df, periods=periods)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    FORECASTS_CHRONOS_ES_DIR.mkdir(parents=True, exist_ok=True)
    fc_csv = FORECASTS_CHRONOS_ES_DIR / f"forecast_chronos_es_{safe_entity}_{resolution}.csv"
    forecast_df.to_csv(fc_csv, index=False)

    print(f"\n{'='*65}\n  AMAZON CHRONOS-2 ES FORECAST ({resolution} | {resolved.upper()})\n{'='*65}")
    print(f"  Forecast CSV  : {fc_csv}")
    print(f"  Rows Projected: {len(forecast_df)}")
    print(f"{'='*65}\n")


def cmd_chronos_es_evaluate(args: argparse.Namespace) -> None:
    """Evaluate Chronos-2 ES model on holdout test data."""
    resolution = getattr(args, "resolution", "1min")
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    df = _load_es_df(args)
    holdout = getattr(args, "holdout", None)

    forecaster = ESChronosForecaster(resolution=resolution, device=device, multivariate=multivariate)
    forecaster.fit(df)
    res = forecaster.evaluate(df, test_steps=holdout)
    print(f"\n{'='*65}\n  CHRONOS-2 ES EVALUATION RESULTS ({resolution})\n{'='*65}")
    for k, v in res.items():
        print(f"  {k:<12}: {v}")
    print(f"{'='*65}\n")


def cmd_chronos_es_capacity(args: argparse.Namespace) -> None:
    """Run 30-day capacity forecast with Amazon Chronos-2 and produce visual charts."""
    args.resolution = "1min"
    device = getattr(args, "device", CHRONOS_DEVICE)
    multivariate = not getattr(args, "univariate", False)
    resolved = resolve_device(device)
    df = _load_es_df(args)

    forecaster = ESChronosForecaster(resolution="1min", device=device, multivariate=multivariate)
    forecaster.fit(df)

    forecast_days = getattr(args, "forecast_days", 30)
    capacity = forecaster.predict_capacity(df, forecast_days=forecast_days)

    safe_entity = re.sub(r"[^\w\-]", "_", getattr(args, "entity", ES_ENTITY_ID))
    cap_dir = FORECASTS_CHRONOS_ES_DIR / "capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)

    paths = forecaster.save_capacity_forecast(capacity, entity_id=safe_entity, output_dir=cap_dir)
    charts = forecaster.plot_capacity_forecast(
        df,
        capacity,
        entity_id=getattr(args, "entity", ES_ENTITY_ID),
        output_dir=cap_dir,
        forecast_days=forecast_days,
    )

    mode_str = "Multivariate" if multivariate else "Univariate"
    print(f"\n{'='*70}\n  CHRONOS-2 30-DAY CAPACITY FORECAST COMPLETE ({mode_str} | {resolved.upper()})\n{'='*70}")
    if "hourly" in paths:
        print(f"  Hourly CSV   : {paths['hourly'][0]}")
        print(f"  Hourly Excel : {paths['hourly'][1]}")
    if "daily" in paths:
        print(f"  Daily CSV    : {paths['daily'][0]}")
        print(f"  Daily Excel  : {paths['daily'][1]}")
    if "hourly" in charts:
        print(f"  Hourly Chart : {charts['hourly']}")
    if "daily" in charts:
        print(f"  Daily Chart  : {charts['daily']}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Parser Setup
# ---------------------------------------------------------------------------

def build_chronos_parser(subparsers=None) -> argparse.ArgumentParser:
    """Build Amazon Chronos-2 CLI parser."""
    if subparsers is not None:
        p = subparsers.add_parser("chronos", help="Amazon Chronos-2 foundation model pipeline")
    else:
        p = argparse.ArgumentParser(
            prog="main_chronos.py",
            description="Amazon Chronos-2 CLI (CSV & Elasticsearch | Universal / Multivariate | CPU/GPU ready)",
        )

    pipe_subs = p.add_subparsers(dest="pipeline", required=True)

    # --- CSV Subcommands ---
    csv_p = pipe_subs.add_parser("csv", help="Chronos-2 on CSV/Excel host metrics")
    csv_subs = csv_p.add_subparsers(dest="command", required=True)

    p_fit = csv_subs.add_parser("fit", help="Initialize/warm up Chronos-2 for hosts")
    p_fit.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_fit.add_argument("--all", action="store_true", help="Initialize for all hosts")
    p_fit.add_argument("--device", default=CHRONOS_DEVICE, choices=["auto", "cpu", "cuda"], help="Hardware target (default: auto)")
    p_fit.add_argument("--univariate", action="store_true", help="Disable multivariate cross-learning")
    p_fit.set_defaults(func=cmd_chronos_csv_fit)

    p_pred = csv_subs.add_parser("predict", help="Generate zero-shot forecasts and charts")
    p_pred.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_pred.add_argument("--all", action="store_true", help="Predict all hosts")
    p_pred.add_argument("--periods", type=int, default=CHRONOS_HORIZON_CSV, help="Forecast horizon (days)")
    p_pred.add_argument("--device", default=CHRONOS_DEVICE, choices=["auto", "cpu", "cuda"], help="Hardware target (default: auto)")
    p_pred.add_argument("--univariate", action="store_true", help="Disable multivariate cross-learning")
    p_pred.set_defaults(func=cmd_chronos_csv_predict)

    p_eval = csv_subs.add_parser("evaluate", help="Evaluate holdout error metrics (MAE, RMSE, WAPE, MAPE)")
    p_eval.add_argument("--host", default="HYDUPINTAPP16", choices=list(HOSTS.keys()), help="Target host")
    p_eval.add_argument("--all", action="store_true", help="Evaluate all hosts")
    p_eval.add_argument("--holdout", type=int, default=CHRONOS_HOLDOUT_DAYS_CSV, help="Holdout days")
    p_eval.add_argument("--device", default=CHRONOS_DEVICE, choices=["auto", "cpu", "cuda"], help="Hardware target (default: auto)")
    p_eval.add_argument("--univariate", action="store_true", help="Disable multivariate cross-learning")
    p_eval.set_defaults(func=cmd_chronos_csv_evaluate)

    # --- ES Subcommands ---
    es_p = pipe_subs.add_parser("es", help="Chronos-2 on Elasticsearch metrics")
    es_subs = es_p.add_subparsers(dest="command", required=True)

    for p_name, func in [
        ("fit", cmd_chronos_es_fit),
        ("predict", cmd_chronos_es_predict),
        ("evaluate", cmd_chronos_es_evaluate),
        ("capacity", cmd_chronos_es_capacity),
    ]:
        sp = es_subs.add_parser(p_name, help=f"{p_name.capitalize()} with Chronos-2 on ES data")
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
        sp.add_argument("--device", default=CHRONOS_DEVICE, choices=["auto", "cpu", "cuda"], help="Hardware target (default: auto)")
        sp.add_argument("--univariate", action="store_true", help="Disable multivariate cross-learning")
        if p_name in ["predict", "evaluate"]:
            sp.add_argument("--periods", type=int, default=None, help="Steps to forecast")
        if p_name in ["evaluate"]:
            sp.add_argument("--holdout", type=int, default=None, help="Holdout steps")
        if p_name == "capacity":
            sp.add_argument("--forecast-days", type=int, default=30, help="Days to forecast")
        sp.set_defaults(func=func)

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_chronos_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
