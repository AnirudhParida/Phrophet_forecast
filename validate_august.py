import pandas as pd
from pipeline.forecaster import ServerMetricsForecaster
from pipeline.visualization import plot_forecast, extract_forecast_rows
from pipeline.evaluation import compute_metrics, format_evaluation_table
from config.settings import FORECASTS_DIR

def main():
    print("Loading data...")
    forecaster = ServerMetricsForecaster(n_forecasts=30)
    all_data = forecaster.load_and_preprocess()
    full_df = all_data['HYDUPINTAPP16'].copy()

    # Split data: train up to 2026-07-31
    train_df = full_df[full_df['ds'] <= '2026-07-31'].copy()
    
    # Replace the internal data with the truncated one
    forecaster._host_data['HYDUPINTAPP16'] = train_df

    print(f"Full data shape: {full_df.shape}")
    print(f"Train data shape: {train_df.shape}")

    print("Training on data up to 2026-07-31...")
    forecaster.fit('HYDUPINTAPP16')

    print("Predicting August 2026...")
    # predict() will forecast 30 days starting from 2026-08-01
    forecasts = forecaster.predict('HYDUPINTAPP16', save_charts=False)

    eval_results = {}

    for metric, f_df in forecasts.items():
        print(f"Saving validation chart for {metric}...")
        plot_forecast(
            host_alias='HYDUPINTAPP16_Aug_Validation',
            metric=metric,
            actuals_df=full_df[['ds', metric]],  # FULL DATA so we see actuals in August
            forecast_df=f_df,
            n_lags=forecaster._resolve_n_lags(metric),
            save=True
        )

        # Compute error metrics for the August validation period
        clean_df = extract_forecast_rows(f_df, metric)
        if clean_df is not None:
            # Merge forecasted yhat with actual metric values for matching dates
            merged = pd.merge(clean_df, full_df[['ds', metric]], on='ds', how='inner')
            if not merged.empty:
                actuals = merged[metric].values
                predicted = merged['yhat'].values
                eval_results[metric] = compute_metrics(actuals, predicted, metric)

    if eval_results:
        # Print the formatted evaluation table
        print(format_evaluation_table(eval_results, 'HYDUPINTAPP16_Aug_Validation', "mixed"))

if __name__ == '__main__':
    main()
