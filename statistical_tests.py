from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Augmented Dickey-Fuller Test
def adf_test_all_columns(df, columns):
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
def plot_stl_decomposition(df, columns, seasonal_period = 12):
    """
    Perform STL decomposition on multiple time series columns and visualize the components.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - columns (list): List of column names to perform STL decomposition on.
    - seasonal_period (int): The seasonal period of the time series. Default is 12 for monthly data.
    """
    for column in columns:

        # Perform STL decomposition
        stl = STL(df[column], seasonal_period)
        result = stl.fit()

        # Plot the decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"STL Decomposition for {column}", fontsize=14)

        axes[0].plot(df.index, df[column], label="Original")
        axes[0].set_title("Original Time Series")

        axes[1].plot(df.index, result.trend, label="Trend", color="green")
        axes[1].set_title("Trend Component")

        axes[2].plot(df.index, result.seasonal, label="Seasonality", color="orange")
        axes[2].set_title("Seasonal Component")

        axes[3].plot(df.index, result.resid, label="Residuals", color="red")
        axes[3].set_title("Residuals (Noise)")

        plt.show()
