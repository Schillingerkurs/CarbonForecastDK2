# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:35:29 2024

@version: 1.0
@author: fs.egb
"""


#import pandas as pd
from pathlib import Path
#from prophet import Prophet
import pickle

# Define the base directory
HERE = Path(__file__).parent.parent.parent


import pandas as pd
from pathlib import Path
from prophet import Prophet
import pickle

# Define the base directory
HERE = Path(__file__).parent.parent.parent

def load_model(model_path):
    """
    Loads the Prophet model from a pickle file.

    Args:
        model_path (Path): The path to the pickle file containing the model.

    Returns:
        Prophet: The loaded Prophet model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_forecast(model, periods=24, freq='H', regressors=None):
    """
    Makes a forecast for the specified number of periods.

    Args:
        model (Prophet): The trained Prophet model.
        periods (int): Number of periods to forecast.
        freq (str): Frequency of the forecast periods.
        regressors (pd.DataFrame): DataFrame containing the regressor values for the forecast period.
        ( i.e. 'latest_forecasted_wind_y_avg', latest_forecasted_production_avg'
         and 'latest_forecasted_power_net_import_DE_avg' in the working model)

    Returns:
        pd.DataFrame: The forecasted DataFrame.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)

    if regressors is not None:
        future = future.merge(regressors, on='ds', how='left')

    forecast = model.predict(future)
    return forecast

def main(future_df):
    # Load the model
    model_path = HERE / "models" / 'prophet_model_with_wind.pkl'
    model = load_model(model_path)

    # Make a forecast for the next 24 hours
    forecast = make_forecast(model, periods=24, freq='H', regressors = future_df)

    # Print the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))

    # Optionally, save the forecast to a CSV file
    forecast.to_csv(HERE / "data" / "output" / "forecast_24h.csv", index=False)

    # Plot the forecast
    fig = model.plot(forecast)
    fig.show()

if __name__ == "__main__":
    main()