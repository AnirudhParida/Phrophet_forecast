"""
main_chronos.py
===============
Direct CLI entrypoint for Amazon Chronos-2 universal foundation model forecasting pipeline
across CSV and Elasticsearch data.

Supports:
- Multivariate Group Attention across all host metrics (--multivariate vs --univariate)
- Local CPU inference and GPU acceleration with zero code changes:
  --device auto   (default: auto-detects CUDA GPU if present, otherwise uses CPU)
  --device cpu    (forces CPU execution)
  --device cuda   (forces CUDA GPU execution with fallback check)

Examples:
  # CSV host forecasting (Multivariate default)
  python main_chronos.py csv fit --host HYDUPINTAPP16
  python main_chronos.py csv predict --host HYDUPINTAPP16 --periods 90
  python main_chronos.py csv evaluate --host HYDUPINTAPP16 --holdout 30

  # CSV host forecasting (Univariate mode)
  python main_chronos.py csv predict --host HYDUPINTAPP16 --univariate

  # Elasticsearch streaming metrics
  python main_chronos.py es fit --resolution 1h
  python main_chronos.py es predict --resolution 1h --periods 24
  python main_chronos.py es evaluate --resolution 1h --holdout 24
  python main_chronos.py es capacity --forecast-days 30
"""

from cli.cli_chronos import main

if __name__ == "__main__":
    main()
