"""
main_tfm.py
===========
Direct CLI entrypoint for Google TimesFM 3.0 foundation model forecasting pipeline
across CSV and Elasticsearch data.

Supports local CPU inference and GPU acceleration with zero code changes:
  --device auto   (default: auto-detects CUDA GPU if present, otherwise uses CPU)
  --device cpu    (forces CPU execution)
  --device cuda   (forces CUDA GPU execution with fallback check)

Examples:
  # CSV host forecasting
  python main_tfm.py csv fit --host HYDUPINTAPP16
  python main_tfm.py csv predict --host HYDUPINTAPP16 --periods 90
  python main_tfm.py csv evaluate --host HYDUPINTAPP16 --holdout 30

  # Elasticsearch streaming metrics
  python main_tfm.py es fit --resolution 1h
  python main_tfm.py es predict --resolution 1h --periods 24
  python main_tfm.py es evaluate --resolution 1h --holdout 24
  python main_tfm.py es capacity --forecast-days 30
"""

from cli.cli_timesfm import main

if __name__ == "__main__":
    main()
