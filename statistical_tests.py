from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Augmented Dickey-Fuller Test
def adf_test(df, columns):
    results = {}
    for col in columns:  
        result = adfuller(df[col], autolag='AIC')
        rounded_crit_values = {key: round(value, 3) for key, value in result[4].items()}  # Round to 3 decimals
        results[col] = {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Values": rounded_crit_values,
            "Stationary": "Yes" if result[1] < 0.05 else "No"
        }
    
    return pd.DataFrame(results).T


# Granger Causality Test
def granger_causality_test(data, target_column, max_lag=4, threshold=0.05):
    """
    Performs Granger Causality Test for every column in the dataset with respect to a target column.
    
    Parameters:
    - data: DataFrame containing the time series data.
    - target_column: The column to test Granger causality against.
    - max_lag: The maximum lag to consider for the test (default is 4).
    - threshold: The p-value threshold to classify causality ("Yes" or "No").
    
    Returns:
    - DataFrame: A DataFrame with Granger causality results for each column.
    """
    granger_results = {}

    for column in data.columns:
        if column != target_column:
            # Perform Granger causality test between column and target column
            test_result = grangercausalitytests(data[[column, target_column]], max_lag, verbose=False)

            # Initialize dictionary to store results ("Yes" or "No" based on p-value)
            causality_results = {}

            # Loop over the lags and check p-value
            for lag, test_results in test_result.items():
                # Access p-value for the SSR chi-square test (or another test)
                p_value = test_results[0]['ssr_chi2test'][1]  # Extract p-value from the chi-squared test
                
                # If p-value is below threshold, store "Yes", else "No"
                causality_results[lag] = "Yes" if p_value < threshold else "No"
            
            # Store the results for the current column
            granger_results[column] = causality_results

    # Convert the results to a DataFrame for easier interpretation
    granger_df = pd.DataFrame(granger_results)

    return granger_df


# STL Decomposition
def stl_decomposition(df, columns, seasonal_period=12):
    """
    Perform STL decomposition on multiple time series columns and visualize the components using Plotly.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - columns (list): List of column names to perform STL decomposition on.
    - seasonal_period (int): The seasonal period of the time series. Default is 12 for monthly data.
    """
    for column in columns:
        # Perform STL decomposition
        stl = STL(df[column], seasonal_period)
        result = stl.fit()

        # Create subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            subplot_titles=("Original Time Series", "Trend Component", "Seasonal Component", "Residuals"))

        # Add traces for each component
        fig.add_trace(go.Scatter(x=df.index, y=df[column], name="Original", line=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.trend, name="Trend", line=dict(color="green")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, name="Seasonality", line=dict(color="orange")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=result.resid, name="Residuals", line=dict(color="red")), row=4, col=1)

        # Update layout
        fig.update_layout(title_text=f"STL Decomposition for {column}", height=800, showlegend=False)
  
        fig.update_xaxes(title_text="Time", row=4, col=1)  # Only add x-axis label to the last row

        # Show the figure
        fig.show()
