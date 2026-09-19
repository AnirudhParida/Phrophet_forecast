"""
main.py
=======
Unified Master CLI for the Infrastructure Forecasting Pipeline.

Supports:
1. NeuralProphet commands : `python main.py np {csv, es} ...`
2. Holt-Winters commands  : `python main.py hw {csv, es} ...`
3. Model comparison       : `python main.py compare {csv, es} ...`
4. Legacy commands        : `python main.py {inspect, fit, predict, evaluate, compare-lags}`

Quick Examples
--------------
  # Unified syntax:
  python main.py np csv fit --host HYDUPINTAPP16
  python main.py hw csv fit --host HYDUPINTAPP16
  python main.py np es fit --resolution 1h
  python main.py hw es fit --resolution 1h
  python main.py compare csv --host HYDUPINTAPP16 --holdout 30
  python main.py compare es --resolution 1h --holdout 24

  # Legacy syntax (backward compatible):
  python main.py inspect
  python main.py fit --host HYDUPINTAPP16 --n-lags 7
  python main.py predict --host HYDUPINTAPP16
  python main.py evaluate --host HYDUPINTAPP16 --holdout 30
  python main.py compare-lags --host HYDUPINTAPP16
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from cli.cli_compare import build_compare_parser
from cli.cli_holtwinters import build_hw_parser, cmd_hw_csv_evaluate, cmd_hw_csv_fit, cmd_hw_csv_predict
from cli.cli_neuralprophet import build_np_parser, cmd_np_csv_compare_lags, cmd_np_csv_evaluate, cmd_np_csv_fit, cmd_np_csv_predict
from cli.cli_timesfm import build_tfm_parser
from cli.cli_chronos import build_chronos_parser
from config.settings import HOLDOUT_DAYS, HOSTS, N_FORECASTS
from pipeline.ingestion import describe_dataset, load_and_preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Legacy Command Handlers
# ---------------------------------------------------------------------------

def _legacy_inspect(_args: argparse.Namespace) -> None:
    print("\n🔍 Loading and inspecting dataset …")
    host_data = load_and_preprocess()
    describe_dataset(host_data)


def _legacy_fit(args: argparse.Namespace) -> None:
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        cmd_hw_csv_fit(args)
    else:
        cmd_np_csv_fit(args)


def _legacy_predict(args: argparse.Namespace) -> None:
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        cmd_hw_csv_predict(args)
    else:
        cmd_np_csv_predict(args)


def _legacy_evaluate(args: argparse.Namespace) -> None:
    if getattr(args, "model", "neuralprophet") in ["holt-winters", "hw"]:
        cmd_hw_csv_evaluate(args)
    else:
        cmd_np_csv_evaluate(args)


def _legacy_compare_lags(args: argparse.Namespace) -> None:
    cmd_np_csv_compare_lags(args)


def _legacy_compare_models(args: argparse.Namespace) -> None:
    from cli.cli_compare import cmd_compare_csv
    cmd_compare_csv(args)


# ---------------------------------------------------------------------------
# Argument Parser Setup
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified Infrastructure Metrics Forecasting Pipeline (NeuralProphet & Holt-Winters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Modular Subparsers
    build_np_parser(subparsers)
    build_hw_parser(subparsers)
    build_tfm_parser(subparsers)
    build_chronos_parser(subparsers)
    build_compare_parser(subparsers)

    # 2. Legacy Subparsers for 100% Backward Compatibility
    # --- inspect ---
    subparsers.add_parser("inspect", help="Print dataset summary — no training").set_defaults(func=_legacy_inspect)

    # --- fit ---
    p_fit = subparsers.add_parser("fit", help="Train model for a host (or all hosts)")
    p_fit.add_argument("--host", choices=list(HOSTS.keys()), help="Host alias")
    p_fit.add_argument("--all", action="store_true", help="Train models for all hosts")
    p_fit.add_argument("--parallel", action="store_true", help="Fit hosts in parallel")
    p_fit.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_fit.add_argument("--model", default="neuralprophet", choices=["neuralprophet", "holt-winters", "hw"])
    p_fit.set_defaults(func=_legacy_fit)

    # --- predict ---
    p_pred = subparsers.add_parser("predict", help="Generate forecast charts")
    p_pred.add_argument("--host", required=True, choices=list(HOSTS.keys()), help="Host alias")
    p_pred.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_pred.add_argument("--periods", type=int, default=N_FORECASTS, help="Forecast horizon (days)")
    p_pred.add_argument("--model", default="neuralprophet", choices=["neuralprophet", "holt-winters", "hw"])
    p_pred.set_defaults(func=_legacy_predict)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model accuracy on holdout set")
    p_eval.add_argument("--host", required=True, choices=list(HOSTS.keys()), help="Host alias")
    p_eval.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Trailing holdout days")
    p_eval.add_argument("--n-lags", type=int, choices=[7, 14], default=None, help="AR lookback lags")
    p_eval.add_argument("--model", default="neuralprophet", choices=["neuralprophet", "holt-winters", "hw"])
    p_eval.set_defaults(func=_legacy_evaluate)

    # --- compare-lags ---
    p_lag = subparsers.add_parser("compare-lags", help="Compare n_lags=7 vs n_lags=14")
    p_lag.add_argument("--host", required=True, choices=list(HOSTS.keys()), help="Host alias")
    p_lag.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout window")
    p_lag.set_defaults(func=_legacy_compare_lags)

    # --- compare-models ---
    p_cmp = subparsers.add_parser("compare-models", help="Compare NeuralProphet vs Holt-Winters side-by-side")
    p_cmp.add_argument("--host", required=True, choices=list(HOSTS.keys()), help="Host alias")
    p_cmp.add_argument("--holdout", type=int, default=HOLDOUT_DAYS, help="Holdout window")
    p_cmp.set_defaults(func=_legacy_compare_models)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user.")
            sys.exit(130)
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
