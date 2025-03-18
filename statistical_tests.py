from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

def outliers_stationarity(df):
    """
    Performs stationarity analysis on a time series DataFrame.
    
    For each column:
    - Conducts the Augmented Dickey-Fuller (ADF) test.
    - Plots the original series with Plotly.
    - Plots the Autocorrelation Function (ACF) with Matplotlib.
    - Plots the Partial Autocorrelation Function (PACF) with Matplotlib.
    """

    for column in df.columns:
        # Perform ADF test
        result = adfuller(df[column].dropna())  # Drop NaN values for ADF test
        adf_statistic, p_value = result[0], result[1]

        # Print results
        print(f'Results for {column}: | ADF Statistic: {adf_statistic:.6f} | p-value: {p_value:.6f}')

        # ---- PLOTLY Time Series Plot ----
        fig_ts = go.Figure()

        fig_ts.add_trace(go.Scatter(
            x=df.index, 
            y=df[column], 
            mode='lines', 
            name=column,
            line=dict(color='blue')
        ))

        fig_ts.update_layout(
            title=f"ADF test for ({column})",
            xaxis_title="Time",
            yaxis_title="Value",
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(showgrid=True),
            template="plotly_white"
        )

        fig_ts.show()

        # Determine number of lags dynamically
        n = len(df[column].dropna())
        lags = max(1, min(n // 4, 40))

        # ---- Matplotlib for ACF & PACF ----
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        # ACF Plot
        plot_acf(df[column], ax=axes[0], lags=lags)
        axes[0].set_title(f'Autocorrelation (ACF) ({column})')

        # PACF Plot
        plot_pacf(df[column], ax=axes[1], lags=lags)
        axes[1].set_title(f'Partial Autocorrelation (PACF) ({column})')

        plt.show()
