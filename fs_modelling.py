from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import mutual_info_regression

import seaborn as sns
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.base import clone
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb


# Train-Validation Split
def train_val_split(df, target_cols, val_percentage):
    """
    Splits a time series dataframe into training and validation sets based on a percentage of validation data.
    
    Args:
        df (pd.DataFrame): The time series dataframe with a DateTime index.
        target_cols (str or list): The name of the target column or a list of target columns.
        val_percentage (float): The percentage of the data to use for validation (between 0 and 1).
        
    Returns:
        X_train, X_val, y_train, y_val (pd.DataFrames)
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    n_obs = len(df)
    val_size = int(n_obs * val_percentage)
    split_index = n_obs - val_size
    
    train = df.iloc[:split_index]
    val = df.iloc[split_index:]
    
    X_train = train.drop(columns=target_cols)
    y_train = train[target_cols]
    
    X_val = val.drop(columns=target_cols)
    y_val = val[target_cols]
    
    return X_train, X_val, y_train, y_val


# Scaling Data
def scale_data(X_train, X_val, scaler_type='minmax'):
    """
    Scales the training and validation data using the specified scaler.
    
    Parameters:
        X_train (pd.DataFrame): The training features.
        X_val (pd.DataFrame): The validation features.
        scaler_type (str): The type of scaler to use.
    
    Returns:
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_val_scaled (pd.DataFrame): Scaled validation features.
    """
    # Choose scaler based on input
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type. Choose either 'minmax' or 'standard'.")
    
    # Fit on training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform the validation data using the same scaler
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    return X_train_scaled, X_val_scaled

# Feature Selection
def correlation(X_train, corr_threshold=0.85, plot=False):
    corr_matrix = X_train.corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features = {column for column in upper_triangle.columns if any(upper_triangle[column].abs() > corr_threshold)}
    selected_features = [col for col in X_train.columns if col not in correlated_features]
    
    print(f'Selected {len(selected_features)} features by correlation')

    if plot:
        plt.figure(figsize=(12, 8))

        # Mask the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create a heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)

        plt.title(f'Feature Correlation Matrix (Threshold = {corr_threshold})')
        plt.show()

    return selected_features

def rfe(X_train, y_train, rfe_model=None, plot=False):
    if rfe_model is None:
        rfe_model = LinearRegression()
    
    rfecv = RFECV(estimator=rfe_model, cv=5, 
                  min_features_to_select=50, scoring='neg_root_mean_squared_error')
    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_].tolist()
    print(f'Selected {len(selected_features)} features by RFECV')

    if plot:
        num_features = np.arange(1, len(rfecv.cv_results_['mean_test_score'])+ 1)
        rmse_scores = np.abs(rfecv.cv_results_['mean_test_score'])

        plt.figure(figsize=(10, 5))
        plt.plot(num_features, rmse_scores, marker='o', linestyle='-')
        plt.axvline(x=len(selected_features), color='r', linestyle='--', label=f'Optimal: {len(selected_features)} features')
        plt.xlabel("Number of Features Selected")
        plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.title("RFECV: Number of Features vs. RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()

    return selected_features

def feature_importance(X_train, y_train, importance_threshold='mean', plot=False):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    importances = model.feature_importances_

    # Determine the threshold value
    if importance_threshold == 'mean':
        threshold_value = np.mean(importances)
    elif importance_threshold == 'median':
        threshold_value = np.median(importances)
    else:
        threshold_value = float(importance_threshold)

    selected_features = X_train.columns[importances > threshold_value].tolist()
    print(f'Selected {len(selected_features)} features by importance with threshold {threshold_value}')

    if plot:
        plt.figure(figsize=(12, 6))

        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]  
        sorted_features = X_train.columns[sorted_indices]
        sorted_importances = importances[sorted_indices]

        plt.bar(sorted_features[:20], sorted_importances[:20]) 
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Feature Importance Score")
        plt.title("Feature Importance Based on Random Forest")
        plt.show()

    return selected_features

def lasso_(X_train, y_train, plot=False):
    lasso = LassoCV(cv=10, random_state=42).fit(X_train, y_train)
    selected_features = X_train.columns[lasso.coef_ != 0].tolist()
    
    print(f'Selected {len(selected_features)} features by Lasso regularization')

    if plot:
        plt.figure(figsize=(12, 6))

        # Sort features by absolute coefficient values
        coef_values = lasso.coef_
        sorted_indices = np.argsort(np.abs(coef_values))[::-1]  # Descending order
        sorted_features = X_train.columns[sorted_indices]
        sorted_coefs = coef_values[sorted_indices]

        plt.bar(sorted_features[:20], sorted_coefs[:20])  # Plot top 20 features
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Lasso Coefficients")
        plt.title("Feature Importance Based on Lasso Regularization")
        plt.show()

    return selected_features

def mutual_info(X_train, y_train, plot=False, mi_threshold=0.1):

    # Compute mutual information between each feature and the target
    mutual_info_scores = mutual_info_regression(X_train, y_train)
    
    # Create a DataFrame to view feature importance
    mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information': mutual_info_scores})
    
    # Sort the features by mutual information score
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
    
    # Select features above the threshold
    selected_features = mi_df[mi_df['Mutual Information'] >= mi_threshold]['Feature'].tolist()
    
    print(f'Selected {len(selected_features)} features by Mutual Information')

    if plot:
        # Plot the mutual information scores for each feature
        plt.figure(figsize=(10, 6))
        plt.bar(mi_df['Feature'], mi_df['Mutual Information'])
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Mutual Information")
        plt.title("Mutual Information of Features with Target")
        plt.show()

    return selected_features

def feature_selection(X_train, y_train, method='all', rfe_model=None, 
                      corr_threshold=0.85, importance_threshold='mean', 
                      mi_threshold=0.1, plot=False):
    if method == 'correlation':
        return correlation(X_train, corr_threshold, plot)
    elif method == 'rfe':
        return rfe(X_train, y_train, rfe_model, plot)
    elif method == 'importance':
        return feature_importance(X_train, y_train, importance_threshold, plot)
    elif method == 'lasso':
        return lasso_(X_train, y_train, plot)
    elif method == 'mutual_info':
        return mutual_info(X_train, y_train, plot=False, mi_threshold=mi_threshold)
        
    elif method == 'all':
        corr_features = set(correlation(X_train, corr_threshold, plot))
        rfe_features = set(rfe(X_train, y_train, rfe_model, plot))
        importance_features = set(feature_importance(X_train, y_train, importance_threshold, plot))
       # lasso_features = set(lasso_(X_train, y_train, plot)) --> this is purpurposefully commented out
        mi_features = set(mutual_info(X_train, y_train, plot=False, mi_threshold=mi_threshold))

        selected_features = list(corr_features & rfe_features & importance_features & mi_features)
        print(f'Selected {len(selected_features)} features that intersect across all methods')
        return selected_features
        
    else:
        raise ValueError("Invalid method. Choose from 'correlation', 'rfe', 'importance', 'lasso', or 'all'.")
    


# Statistical Models
def stats_models(model_type, X_train, X_val, y_train, y_val,
                 order=(1,1,1), seasonal_order=(1,1,1,12), plot=False, csv_path=None):

    """
    Train and evaluate time series forecasting models (ARIMA, SARIMAX, Prophet).
    
    Parameters:
    ----------
    model_type : str
        The type of model to use. Options: 'arima', 'sarimax', or 'prophet'.
    X_train : pd.DataFrame or None
        Training features (only used for SARIMAX, ignored for ARIMA and Prophet).
    X_val : pd.DataFrame or None
        Validation features (only used for SARIMAX, ignored for ARIMA and Prophet).
    y_train : pd.Series
        Training target variable (time series data).
    y_val : pd.Series
        Validation target variable (time series data).
    order : tuple, default=(1,1,1)
        ARIMA/SARIMAX model parameters (p, d, q) for non-seasonal components.
    seasonal_order : tuple, default=(1,1,1,12)
        SARIMAX model parameters (P, D, Q, S) for seasonal components.
    plot : bool, default=False
        If True, plots the actual vs. predicted values for the validation set.
    csv_path : str or None, default=None
        If specified, appends model performance metrics to the given CSV file.
    
    Returns:
    -------
    model_fit : statsmodels
        The trained model object.
    val_preds : pd.Series
        Predictions for the validation set.
    summary : statsmodels summary object or pd.Series
        Model summary for ARIMA/SARIMAX.
    """

    train_rmse, val_rmse, train_mape, val_mape = None, None, None, None  

    if model_type == 'arima':
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        train_preds = model_fit.fittedvalues
        val_preds = model_fit.forecast(steps=len(y_val))

    elif model_type == 'sarimax':
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, exog=X_train)
        model_fit = model.fit()
        train_preds = model_fit.fittedvalues
        val_preds = model_fit.forecast(steps=len(y_val), exog=X_val)

    elif model_type == 'prophet':
        df_train = y_train.reset_index()
        df_train.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(df_train)

        future = pd.DataFrame({'ds': y_val.index})
        forecast_df = model.predict(future)
        val_preds = forecast_df.set_index('ds')['yhat']

        train_preds = None
        model_fit = model

    else:
        raise ValueError("Invalid model_type. Choose 'arima', 'sarimax', or 'prophet'.")

    # Calculate Error Metrics
    if train_preds is not None:
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_mape = mean_absolute_percentage_error(y_train, train_preds) * 100

    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mape = mean_absolute_percentage_error(y_val, val_preds) * 100

    # Optional Plot
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train, label='Actual Train', color='blue')
        plt.plot(y_val.index, y_val, label='Actual Validation', color='green')
        plt.plot(y_val.index, val_preds, label='Predicted Validation', color='red')

        plt.legend()
        plt.title(f'{model_type.upper()} Model - Training and Validation Predictions')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.show()

    # Collect results
    results = {
        "model_type": model_type,
        "features_used": X_train.columns.tolist() if model_type == 'sarimax' else "N/A",
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mape (%)": train_mape,
        "val_mape (%)": val_mape
    }

    # Save results to CSV if a path is specified
    if csv_path:
        results_df = pd.DataFrame([results])  # Convert dict to DataFrame
        
        # Check if file exists to determine mode
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, mode='w', header=True, index=False)
        
        print(f"Results appended to {csv_path}")

    if model_type == 'prophet':
        return model_fit, val_preds  
    else:
        return model_fit, val_preds, model_fit.summary()

# All models

def all_models(model, X_train, X_val, target_train, target_val, plot=False, csv_path=None, **model_params):
    """
    Train any regression model and evaluate its performance.
    
    Parameters:
    - model: The regression model instance (e.g., XGBRegressor, RandomForestRegressor).
    - X_train: Training features.
    - X_val: Validation features.
    - target_train: Training target variable.
    - target_val: Validation target variable.
    - plot (bool): Whether to plot actual vs predicted values.
    - csv_path (str, optional): Path to save the results CSV file.
    - **model_params: Additional parameters to pass to the model.
    """
    
    model.set_params(**model_params)
    
    # Fit 
    model.fit(X_train, target_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(target_train, train_preds))
    train_mape = mean_absolute_percentage_error(target_train, train_preds) * 100
    
    val_rmse = np.sqrt(mean_squared_error(target_val, val_preds))  
    val_mape = mean_absolute_percentage_error(target_val, val_preds) * 100 
    
    # Plot
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(target_train.index, target_train, label='Actual Train', color='blue', alpha=0.7)
        plt.plot(target_val.index, target_val, label='Actual Validation', color='green', alpha=0.7)
        plt.plot(target_val.index, val_preds, label='Predicted Validation', color='red')
        plt.legend()
        plt.title(f'{model.__class__.__name__} - Training and Validation Predictions')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.show()

    # Store results
    results = {
        "model_type": model.__class__.__name__,
        "features_used": X_train.columns.tolist(),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mape (%)": train_mape,
        "val_mape (%)": val_mape
    }

    # Save results to CSV if a path is provided
    if csv_path:
        results_df = pd.DataFrame([results])
        
        # Check if file exists to determine mode
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, mode='w', header=True, index=False)
        
        print(f"Results appended to {csv_path}")

    return model, val_preds
