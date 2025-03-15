from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

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
