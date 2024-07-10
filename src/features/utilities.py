# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:37:21 2024

@author: fs.egb
"""

import pandas as pd



import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor



def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_error_metrics(df_test,model):
    test_predictions = model.predict(df_test)
    
    error  = mean_absolute_percentage_error(y_true=df_test['y'],
                   y_pred = test_predictions['yhat'])
    
    print(f"Mean Absolute Percentage Error: {error:.2f}%")
    
    
    


def get_recursive_feature_elimination(df):
  
    # Correlation analysis
    correlation_matrix = df.corr()
    correlations_with_target = correlation_matrix['y'].abs().sort_values(ascending=False)
    
    # Select top N features based on correlation
    top_features = correlations_with_target.head(20).index.tolist()
    
    # RFE with a RandomForestRegressor
    X = df[top_features]
    y = df['y']
    
    # Initialize RandomForestRegressor
    model = RandomForestRegressor()
    
    # Initialize RFE
    rfe = RFE(model, n_features_to_select=10)
    
    # Fit RFE
    rfe = rfe.fit(X, y)
    
    # Get the selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Print the selected features
    print("Selected features:", selected_features)
    
    
    return selected_features

    
    
    
    
def plot_components_and_performance(df_test, model):
    
    test_predictions = model.predict(df_test)

    
    fig_components = model.plot_components(test_predictions)
 
    
    # Plot the forecast with the actuals
    f, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(df_test['ds'], df_test['y'], color='r')
    fig_forecast = model.plot(test_predictions, ax=ax)
    
    return fig_forecast, fig_components


# List of features measured inconsistently
def plot_correlation_with_y(df, target_column='y', filename='correlation_with_y.png'):
    """
    Plots and saves the correlation of each column with a specified target column ('y') in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column to calculate correlations with. Default is 'y'.
    filename (str): The name of the file to save the plot. Default is 'correlation_with_y.png'.

    Returns:
    None
    """
    # Calculate the correlation of each column with 'y'
    correlations = df.corr()[[target_column]].sort_values(by=target_column, ascending=False)

    # Plot the correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    # Customize plot
    plt.title(f'Correlation with {target_column}')
    plt.yticks(rotation=0)  # Ensure y-axis labels are not rotated

    # Save the figure locally
    plt.savefig(filename)

    # Show the plot
    plt.show()