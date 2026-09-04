"""
main.py
=======
Command-line interface for the NeuralProphet infrastructure forecasting pipeline.

Available commands
------------------
  inspect       — Print a dataset summary (no training required)
  fit           — Train NeuralProphet models for one or all hosts
  predict       — Load saved models and generate 7-day forecast charts
  evaluate      — Compute MAE / RMSE / MAPE on a trailing holdout set
  compare-lags  — Train with n_lags=7 AND n_lags=14 and compare results

Quick start (step-by-step)
--------------------------
See README.md for the full ordered training guide.

Example usage
-------------
  python main.py inspect
  python main.py fit --host HYDUPINTAPP16 --n-lags 7
  python main.py predict --host HYDUPINTAPP16 --n-lags 7
  python main.py evaluate --host HYDUPINTAPP16 --holdout 30 --n-lags 7
  python main.py compare-lags --host HYDUPINTAPP16
  python main.py fit --all --n-lags 7
  python main.py fit --all --parallel --n-lags 7
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config.settings import HOLDOUT_DAYS, HOSTS, N_LAGS_DEFAULT, N_FORECASTS
from pipeline import ServerMetricsForecaster, describe_dataset, load_and_preprocess

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "NeuralProphet Infrastructure Metrics Forecasting Pipeline\n"
            "Forecast CPU, Memory, and Disk % for HYDUPINTAPP16 & JPRUPIWEBCRP02"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py inspect\n"
            "  python main.py fit --host HYDUPINTAPP16 --n-lags 7\n"
            "  python main.py predict --host HYDUPINTAPP16 --n-lags 7\n"
            "  python main.py evaluate --host HYDUPINTAPP16 --holdout 30\n"
            "  python main.py compare-lags --host HYDUPINTAPP16\n"
            "  python main.py fit --all --parallel\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- inspect ---
    subparsers.add_parser(
        "inspect",
        help="Print dataset summary (shape, date range, value ranges) — no training",
    )

    # --- fit ---
    fit_parser = subparsers.add_parser(
        "fit",
        help="Train NeuralProphet models for a host (or all hosts)",
    )
    _add_host_arg(fit_parser, required=False)
    fit_parser.add_argument(
        "--all",
        action="store_true",
        help="Train models for all hosts",
    )
    fit_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Fit hosts in parallel using ProcessPoolExecutor (implies --all)",
    )
    _add_nlags_arg(fit_parser)

    # --- predict ---
    predict_parser = subparsers.add_parser(
        "predict",
        help="Load saved models and generate 7-day forecast charts",
    )
    _add_host_arg(predict_parser, required=True)
    _add_nlags_arg(predict_parser)

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model accuracy on a trailing holdout set",
    )
    _add_host_arg(eval_parser, required=True)
    eval_parser.add_argument(
        "--holdout",
        type=int,
        default=HOLDOUT_DAYS,
        metavar="DAYS",
        help=f"Number of trailing days to hold out (default: {HOLDOUT_DAYS})",
    )
    _add_nlags_arg(eval_parser)

    # --- compare-lags ---
    cmp_parser = subparsers.add_parser(
        "compare-lags",
        help="Train with n_lags=7 AND n_lags=14 and print a side-by-side comparison",
    )
    _add_host_arg(cmp_parser, required=True)
    cmp_parser.add_argument(
        "--holdout",
        type=int,
        default=HOLDOUT_DAYS,
        metavar="DAYS",
        help=f"Holdout window for the comparison (default: {HOLDOUT_DAYS})",
    )

    return parser


def _add_host_arg(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument(
        "--host",
        type=str,
        choices=list(HOSTS.keys()),
        required=required,
        metavar="HOST",
        help=f"Host alias: {list(HOSTS.keys())}",
    )


def _add_nlags_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--n-lags",
        type=int,
        default=None,
        choices=[7, 14],
        metavar="N",
        help="AR lookback: 7 or 14 (default: auto-resolves per-metric optimal lookbacks)",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_inspect(_args: argparse.Namespace) -> None:
    """Print a human-readable summary of the raw dataset."""
    print("\n🔍  Loading and inspecting dataset …")
    host_data = load_and_preprocess()
    describe_dataset(host_data)


def cmd_fit(args: argparse.Namespace) -> None:
    """Train models for one or all hosts."""
    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    if args.parallel or args.all:
        host_str = "ALL HOSTS"
    else:
        if not args.host:
            print("❌  Specify --host <HOST> or use --all.  Exiting.")
            sys.exit(1)
        host_str = args.host

    print(f"\n🚀  Fitting models: {host_str}  |  n_lags={args.n_lags}")

    if args.parallel:
        forecaster.fit(host_name=None, n_lags=args.n_lags, parallel=True)
    elif args.all:
        for alias in HOSTS:
            forecaster.fit(host_name=alias, n_lags=args.n_lags)
    else:
        forecaster.fit(host_name=args.host, n_lags=args.n_lags)

    print(f"\n✅  Training complete.  Models saved to models/saved/")


def cmd_predict(args: argparse.Namespace) -> None:
    """Load saved models and generate forecast charts."""
    from pipeline.visualization import extract_forecast_rows

    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    print(f"\n📈  Predicting: {args.host}  |  n_lags={args.n_lags or 'auto'}")
    forecasts = forecaster.predict(host_name=args.host, n_lags=args.n_lags, save_charts=True)

    print(f"\n📋  30-Day Forecast Table (with 90% Confidence Intervals):\n")
    for metric, f_df in forecasts.items():
        clean_df = extract_forecast_rows(f_df)
        if clean_df is not None:
            print(f"--- {args.host} | {metric} ---")
            for idx, row in clean_df.iterrows():
                ds_str = pd.to_datetime(row['ds']).strftime('%Y-%m-%d')
                yhat_str = f"{row['yhat']:6.2f}%"
                lower_str = f"{row.get('yhat_lower', float('nan')):6.2f}%"
                upper_str = f"{row.get('yhat_upper', float('nan')):6.2f}%"
                print(f"  {ds_str}  |  yhat (forecast): {yhat_str}  |  90% CI: [{lower_str} ... {upper_str}]")
            print()



def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run holdout evaluation and print metrics."""
    forecaster = ServerMetricsForecaster(n_lags=args.n_lags, n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    print(
        f"\n📊  Evaluating: {args.host}  |  "
        f"n_lags={args.n_lags}  |  holdout={args.holdout} days"
    )
    forecaster.evaluate(
        host_name=args.host,
        holdout_days=args.holdout,
        n_lags=args.n_lags,
    )


def cmd_compare_lags(args: argparse.Namespace) -> None:
    """Compare n_lags=7 vs n_lags=14 on holdout data."""
    forecaster = ServerMetricsForecaster(n_forecasts=N_FORECASTS)
    forecaster.load_and_preprocess()

    print(f"\n🔬  Comparing n_lags=7 vs n_lags=14 for: {args.host}")
    forecaster.compare_lags(host_name=args.host, holdout_days=args.holdout)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "inspect":      cmd_inspect,
        "fit":          cmd_fit,
        "predict":      cmd_predict,
        "evaluate":     cmd_evaluate,
        "compare-lags": cmd_compare_lags,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
