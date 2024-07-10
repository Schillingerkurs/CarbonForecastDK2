# -*- coding: utf-8 -*-
"""
Script to load, clean, visualize, and save historical data.

Created on Tue Jul  9 09:42:44 2024
@author: fs.egb
"""

import pandas as pd
import os
from dotenv import load_dotenv
import missingno as msno
import matplotlib.pyplot as plt
from pathlib import Path

# Define HERE as parent directory of current working directory
HERE = Path(os.getcwd()).parent.parent

# Load environment variables from .env file
load_dotenv()

# List of features measured inconsistently
bumpy_measured = ['latest_forecasted_dewpoint_avg', 'latest_forecasted_precipitation_avg',
                  'latest_forecasted_solar_avg', 'latest_forecasted_temperature_avg',
                  'latest_forecasted_wind_x_avg', 'latest_forecasted_wind_y_avg']

def load_csv():
    """
    Load historical data from CSV file specified in environment variable HIST_DATA_DK_URL.
    Drop columns with all NaN values and irrelevant columns.
    """
    return (pd.read_csv(os.environ.get("HIST_DATA_DK_URL"))
               .dropna(axis=1, how='all')
               .drop(columns=['zone_name', 'production_sources'])
           )

def clean_core_features(raw):
    """
    Clean and prepare core features of the DataFrame:
    - Convert 'datetime' to datetime format ('ds')
    - Calculate intervals between timestamps ('intervals')
    - Drop rows where 'carbon_intensity_avg' is NaN or intervals are greater than 1 hour
    """
    df = (raw.assign(ds=lambda x: pd.to_datetime(x['datetime']))
              .assign(intervals=lambda x: x.ds.diff())
              .dropna(subset=['carbon_intensity_avg'])
         )
    
    df = df[df['intervals'] <= pd.Timedelta(hours=1)]
    
    return df.drop(columns=['intervals', 'datetime', 'timestamp'])

def compare_nan_values(df, bumpy_measured):
    """
    Compare NaN values in the DataFrame and save visualization plots.
    Plot overall NaN matrix and NaN matrix for bumpy measured features.
    """
    # Plotting the NaN values for all features
    msno.matrix(df)
    plt.savefig(HERE / "reports" / "figures" / 'nan_plot_all_features.png')
    
    # Plotting the NaN values for bumpy measured features
    msno.matrix(df[bumpy_measured])
    plt.savefig(HERE / "reports" / "figures" / 'bumpy_measured_features.png')

def impute_non_bumpy_measures(df, bumpy_measured):
    """
    Impute NaN values for non-bumpy measured features by forward filling and backward filling.
    Merge imputed non-bumpy features with bumpy measured features.
    """
    non_bumpy_imputed = df.drop(columns=bumpy_measured).ffill().bfill()
    
    df_out = pd.merge(non_bumpy_imputed, df[bumpy_measured], 
                      left_index=True, right_index=True, validate="1:1")
    
    return df_out

# Main script execution
if __name__ == "__main__":
    # Load and clean data
    df = (load_csv()
          .pipe(clean_core_features)
         )
    
    # Visualize and compare NaN values
    compare_nan_values(df, bumpy_measured)
    
    # Impute non-bumpy measures and save cleaned DataFrame as pickle file
    df_imputed = impute_non_bumpy_measures(df, bumpy_measured)
    df_imputed.to_pickle(HERE / "data" / "interim" / "clean_hist_panel.pickle")

   











