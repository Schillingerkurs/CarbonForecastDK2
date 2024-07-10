# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:31:41 2024

@author: fs.egb
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
# Function to create time series features
def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    
    I use defintions of time feature such as  summer/winter/spring/fall from a kaggle notebook.
    REF:https://www.kaggle.com/code/robikscube/time-series-forecasting-with-prophet-yt
    """


    cat_type = CategoricalDtype(categories=['Monday','Tuesday',
                                            'Wednesday',
                                            'Thursday','Friday',
                                            'Saturday','Sunday'],
                                ordered=True)
        
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['date_offset'] = (df['date'].dt.month * 100 + df['date'].dt.day - 320) % 1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                          labels=['Spring', 'Summer', 'Fall', 'Winter']
                   )
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekday',
            'season']]
    
    if label:
        y = df[label]
        return X, y
    return X



def examine_the_last_month(df_test, model ):
    
    
    df_test_index = df_test.set_index("ds")
    
    
    test_predictions = model.predict(df_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_test_index.index, df_test_index['y'], color='r')
    fig = model.plot(test_predictions, ax=ax)
    
    
    start_date = datetime.strptime('2019-09-01', '%Y-%m-%d')
    end_date = datetime.strptime('2019-09-28', '%Y-%m-%d')
    
    ax.set_xbound(lower=start_date, upper=end_date)
    ax.set_ylim(0, 400)
    plot = plt.suptitle('September 2019 Forecast vs Actuals')
    
    return fig





def compare_time_features(df, HERE):
    """
    Create time series features to see what is actually useful.
    
    """
    X, y = create_features(df.set_index("ds"), label='y')
    
    features_and_target = pd.concat([X, y], axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=features_and_target.dropna(),
                x='weekday',
                y='y',
                hue='season',
                ax=ax,
                linewidth=1)
    ax.set_title('Carbon intensity avg by Day of Week')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Carbon Intensity avg')
    ax.legend(bbox_to_anchor=(1, 1))
    
    plt.savefig(HERE / "reports" / "figures" / 'time_features_on_carbon_avg.png')
        