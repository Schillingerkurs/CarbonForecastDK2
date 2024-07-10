# -*- coding: utf-8 -*-
"""

Script to create time series features and forecast carbon_intensity_avg using fbprophet.
Created on Tue Jul  9 10:41:19 2024

@author: fs.egb
"""

import pandas as pd
from pathlib import Path
from prophet import Prophet

# Define path to the pickle file


HERE = Path(__file__).parent.parent.parent


from utilities import calculate_error_metrics,\
                      plot_components_and_performance, plot_correlation_with_y

      
from compare_time_features import   compare_time_features, examine_the_last_month



def main():
    # Load cleaned DataFrame from pickle file
    df = (pd.read_pickle(HERE / "data" / "interim" / "clean_hist_panel.pickle")
          # Need to remove the timezone information from 
          .assign(ds = lambda x: x['ds'].dt.tz_localize(None))
          # rename output column following the notation prophet wants.
          .rename(columns={'carbon_intensity_avg':'y'})
          )
    
    # Split the sample to see how go the time features alone predict carbon footprint
    split_date = '1-September-2019'
    df_train = df.loc[df['ds'] <= split_date].copy()
    df_test = df.loc[df['ds']> split_date].copy()
        
    # Forecast with Prophet using only time features.
    
    baseline_model = Prophet()
    baseline_model.fit(df_train[['y',"ds"]])
    
    fig_forecast, fig_components = plot_components_and_performance(df_test, 
                                          baseline_model)
    
    
    
    
    fig_forecast.savefig(HERE / "reports" / "figures" / 'time_features_on_carbon_forecast.png')
    
    fig_components.savefig(HERE / "reports" / "figures" / 'time_features_forecast_components.png')
    
    
    
    last_moth_plot =  examine_the_last_month( df_test, baseline_model)
    
    last_moth_plot.savefig(HERE / "reports" / "figures" / 'september_forcast.png')
    

    
    calculate_error_metrics(df_test, baseline_model)
    

    # The forcast with considering time trends alone performs relatively bad. 37 % off in absolute error percantages
    
    
    # For a better comparison, lets see how weekly trends compare to seasonale effects. 
    compare_time_features(df, HERE)
   
    # Seasonal patterns seem to matter the most.
    
    
    # Let's look at the forcast  variables and see what correlates with our outcome
      

        # List of features measured inconsistently   
    plot_correlation_with_y(df[['latest_forecasted_price_avg', 'latest_forecasted_production_avg',
                                'latest_forecasted_consumption_avg',
                                'latest_forecasted_power_net_import_DE_avg',
                                'latest_forecasted_power_net_import_DK-DK1_avg',
                                'latest_forecasted_power_net_import_SE-SE4_avg',
                                'latest_forecasted_production_solar_avg',
                                'latest_forecasted_production_wind_avg',
                                "y"]] ,
                            filename = HERE / "reports" / "corr_tables" / 'forecasted_non_bumpy_measures.png')
    
    
    # Best "non-bumpy" predictionrs
    #'latest_forecasted_production_avg', 
    #'latest_forecasted_power_net_import_DE_avg', 
    #'latest_forecasted_consumption_avg',
    
    
    bumpy_measured = ['latest_forecasted_dewpoint_avg', 'latest_forecasted_precipitation_avg',
                      'latest_forecasted_solar_avg', 'latest_forecasted_temperature_avg',
                      'latest_forecasted_wind_x_avg', 'latest_forecasted_wind_y_avg',
                      "y"]
    
    
    bumpy_split = '1-January-2018'
     
     
    df_late = df.loc[df['ds']> bumpy_split].copy()
    
        
    plot_correlation_with_y(df_late[bumpy_measured],
                            filename = HERE / "reports" / "corr_tables" / 'forecasted_bumpy_measures_late.png')
            
    # If we constrain the sample to the later periods (after bumpy split)
    
    # becomes "latest_forecasted_wind_y_avg" a pretty decend estimtor for y.
    
    
    
    model_w_producution = Prophet()
    model_w_producution.add_regressor('latest_forecasted_production_avg')
    model_w_producution.add_regressor('latest_forecasted_power_net_import_DE_avg')
    
    
    # fillna bumpy measure to see if 
    model_w_producution.add_regressor('latest_forecasted_wind_y_avg')
   
    
    df_train['latest_forecasted_wind_y_avg'] = df_train['latest_forecasted_wind_y_avg'].ffill().bfill()
    
    
    model_w_producution.fit(df_train)
    
    
    
    df_test['latest_forecasted_wind_y_avg'] = df_test['latest_forecasted_wind_y_avg'].ffill().bfill()

    
    calculate_error_metrics(df_test, model_w_producution)
    
    

    

    
    future = model.make_future_dataframe(periods=24)
    forecast = model.predict(future)
    
   

# if __name__ == "__main__":
#     main()
