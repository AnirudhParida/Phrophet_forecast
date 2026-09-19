"""
main_hw.py
==========
Direct CLI entrypoint for Holt-Winters Exponential Smoothing forecasting pipeline
across CSV and Elasticsearch data.

Examples:
  python main_hw.py csv fit --host HYDUPINTAPP16
  python main_hw.py csv predict --host HYDUPINTAPP16
  python main_hw.py csv evaluate --host HYDUPINTAPP16 --holdout 30
  python main_hw.py es fit --resolution 1h
  python main_hw.py es predict --resolution 1h
  python main_hw.py es evaluate --resolution 1h --holdout 24
  python main_hw.py es capacity --forecast-days 30
"""

from cli.cli_holtwinters import main

if __name__ == "__main__":
    main()
