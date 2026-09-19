"""
cli/__init__.py
===============
Clean CLI modules for time series forecasting:
- cli_neuralprophet: NeuralProphet CLI (csv, es)
- cli_holtwinters   : Holt-Winters CLI (csv, es)
- cli_compare      : Benchmark & Comparison CLI (csv, es)
"""

from cli.cli_neuralprophet import build_np_parser
from cli.cli_holtwinters import build_hw_parser
from cli.cli_compare import build_compare_parser

__all__ = [
    "build_np_parser",
    "build_hw_parser",
    "build_compare_parser",
]
