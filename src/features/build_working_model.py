

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:40:24 2024

@version: 1.0
@author: fs.egb
"""

import pandas as pd
from pathlib import Path
from prophet import Prophet
import pickle

# Local imports from project utilities
from utilities import calculate_error_metrics, plot_components_and_performance, plot_correlation_with_y
from compare_time_features import compare_time_features, examine_the_last_month

# Define the base directory
HERE = Path(__file__).parent.parent.parent

def drop_until_no_nan(df, column):
    """
    Drops the first row of the DataFrame until there is no NaN in the specified column.

    Args:
        df (pd.DataFrame): The input DataFrame containing time series data.
        column (str): The name of the column to check for NaNs.

    Returns:
        pd.DataFrame: DataFrame with rows dropped until the first row has no NaN in the specified column.
    """
    while pd.isna(df.iloc[0][column]):
        df = df.iloc[1:].reset_index(drop=True)
    return df

def linear_interpolate_nans(df, column):
    """
    Imputes NaNs in a specified column by filling in the linear trend between the first non-NaN
    value before the NaN and the first non-NaN value after the NaN.

    Args:
        df (pd.DataFrame): The input DataFrame containing time series data.
        column (str): The name of the column with NaNs to be imputed.

    Returns:
        pd.DataFrame: DataFrame with NaNs imputed in the specified column.
    """
    df[column] = df[column].interpolate(method='linear', limit_direction='both')
    return df


def main():
    # Load cleaned DataFrame from pickle file
    df = (pd.read_pickle(HERE / "data" / "interim" / "clean_hist_panel.pickle")
          .assign(ds=lambda x: x['ds'].dt.tz_localize(None))
          .rename(columns={'carbon_intensity_avg': 'y'}))
    
    # Process the DataFrame to handle NaNs
    df_wind = (drop_until_no_nan(df, 'latest_forecasted_wind_y_avg')
               [['latest_forecasted_wind_x_avg', 'latest_forecasted_wind_y_avg',
                 'latest_forecasted_production_avg', 'latest_forecasted_power_net_import_DE_avg', 'y', 'ds']])
    
    df_wind = linear_interpolate_nans(df_wind, 'latest_forecasted_wind_y_avg')
    
    # Split the sample into training and testing sets
    split_date = '2019-09-01'
    df_train = df_wind.loc[df_wind['ds'] <= split_date].copy()
    df_test = df_wind.loc[df_wind['ds'] > split_date].copy()
    
    # Initialize and train the Prophet model
    model_with_production = Prophet()
    model_with_production.add_regressor('latest_forecasted_production_avg')
    model_with_production.add_regressor('latest_forecasted_power_net_import_DE_avg')
    model_with_production.add_regressor('latest_forecasted_wind_y_avg')
    
    model_with_production.fit(df_train)
    
    # Calculate error metrics
    calculate_error_metrics(df_test, model_with_production)
    
    # Save the model as a pickle file
    model_path = HERE / "models" / 'prophet_model_with_wind.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_with_production, f)
    
    print(f"Prophet model saved as '{model_path}'")
    



if __name__ == "__main__":
    main()

