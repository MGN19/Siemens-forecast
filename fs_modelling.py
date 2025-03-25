from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd

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
