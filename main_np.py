"""
main_np.py
==========
Direct CLI entrypoint for NeuralProphet forecasting pipeline across CSV and Elasticsearch data.

Examples:
  python main_np.py csv fit --host HYDUPINTAPP16
  python main_np.py csv predict --host HYDUPINTAPP16
  python main_np.py csv evaluate --host HYDUPINTAPP16 --holdout 30
  python main_np.py es fit --resolution 1h
  python main_np.py es predict --resolution 1h
  python main_np.py es evaluate --resolution 1h --holdout 24
  python main_np.py es capacity --forecast-days 30
"""

from cli.cli_neuralprophet import main

if __name__ == "__main__":
    main()
