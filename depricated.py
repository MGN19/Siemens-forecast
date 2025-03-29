
### DEPRICATED
country_data = {
    'China': {
        'Country code': '156',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO156',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP156',
        'Producer Prices Electrical Equipment Code': 'PRI27156_org'
    },
    'France': {
        'Country code': '250',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO250',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP250',
        'Producer Prices Electrical Equipment Code': 'PRI27250_org',
        'Production Electrical Equipment Code': 'PRO27250_org',
        'Production Machinery & Equipment Code': 'PRO28250_org'
    },
    'Germany': {
        'Country code': '276',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO276',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP276',
        'Producer Prices Electrical Equipment Code': 'PRI27276_org',
        'Production Electrical Equipment Code': 'PRO27276_org',
        'Production Machinery & Equipment Code': 'PRO28276_org'
    },
    'Italy': {
        'Country code': '380',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO380',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP380',
        'Producer Prices Electrical Equipment Code': 'PRI27380_org',
        'Production Electrical Equipment Code': 'PRO27380_org',
        'Production Machinery & Equipment Code': 'PRO28380_org'
    },
    'Japan': {
        'Country code': '392',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO392',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP392',
        'Production Electrical Equipment Code': 'PRO27392_org',
        'Production Machinery & Equipment Code': 'PRO28392_org'
    },
    'Switzerland': {
        'Country code': '756',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO756',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP756',
        'Production Electrical Equipment Code': 'PRO27756_org',
        'Production Machinery & Equipment Code': 'PRO28756_org'
    },
    'United Kingdom': {
        'Country code': '826',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO826',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP826',
        'Producer Prices Electrical Equipment Code': 'PRI27826_org',
        'Production Electrical Equipment Code': 'PRO27826_org',
        'Production Machinery & Equipment Code': 'PRO28826_org'
    },
    'United States': {
        'Country code': '840',
        'Production Index Machinery & Electricals': 'MAB_ELE_PRO840',
        'Shipments Index Machinery & Electricals': 'MAB_ELE_SHP840',
        'Producer Prices Electrical Equipment Code': 'PRI27840_org',
        'Production Machinery & Equipment Code': 'PRO28840_org',
        'EUR in LCU': 'WKLWEUR840_org'
    }    
}

# Define the world indicators as another dictionary
world_indicators = {
    'Base Metals Price Index': 'RohiBASEMET1000_org',
    'Energy Price Index': 'RohiENERGY1000_org',
    'Metals & Minerals Price Index': 'RohiMETMIN1000_org',
    'Natural Gas Price Index': 'RohiNATGAS1000_org',
    'Crude Oil Average Price Index': 'RohCRUDE_PETRO1000_org',
    'Copper Price Index': 'RohCOPPER1000_org',
    'Producer Prices Electrical Equipment Code': None,
    'Production Electrical Equipment Code': 'PRO271000_org',
    'Production Machinery & Equipment Code': 'PRO271000_org'
}


colums_name = ['date','CHI_MC_EL_PROD','CHI_MC_EL_SHIP','FRA_MC_EL_PROD','FRA_MC_EL_SHIP','GER_MC_EL_PROD','GER_MC_EL_SHIP','ITA_MC_EL_PROD'
,'ITA_MC_EL_SHIP','JAP_MC_EL_PROD','JAP_MC_EL_SHIP','SWI_MC_EL_PROD','SWI_MC_EL_SHIP','UK_MC_EL_PROD','UK_MC_EL_SHIP'
,'USA_MC_EL_PROD','USA_MC_EL_SHIP','EU_MC_EL_PROD','EU_MC_EL_SHIP','WRL_BASE_METAL_PRICE','WRL_ENERGY_PRICE'
,'WRL_METAL_MINERAL_PRICE','WRL_GAS_PRICE','WRL_AVG_OIL_PRICE','WRL_COPPER_PRICE','USA_EUR_LCU','USA_EE_PRODUCER_PRICE'
,'UK_EE_PRODUCER_PRICE','ITA_EE_PRODUCER_PRICE','FRA_EE_PRODUCER_PRICE','GER_EE_PRODUCER_PRICE','CHI_EE_PRODUCER_PRICE'
,'USA_MC_EQ_PROD','WRL_MC_EQ_PROD','SWI_MC_EQ_PROD','UK_MC_EQ_PROD','ITA_MC_EQ_PROD','JAP_MC_EQ_PROD','FRA_MC_EQ_PROD'
,'GER_MC_EQ_PROD','USA_EE_PROD','WRL_EE_PROD','SWI_EE_PROD','UK_EE_PROD','ITA_EE_PROD','JAP_EE_PROD','FRA_EE_PROD',
 'GER_EE_PROD']


## Models

# Lazy Regressor
def lazy_regressor(X_train, X_val, target_train, target_val, plot = False, csv_path = None):

    # Fit LazyRegressor
    regressor = LazyRegressor(ignore_warnings=True)
    
    lazy_model, lazy_predictions = regressor.fit(X_train, X_val, target_train, target_val)

    best_model_name = lazy_model["RMSE"].idxmin()
    best_model_class = regressor.models[best_model_name]

    best_model = clone(best_model_class)
    best_model.fit(X_train, target_train)

    train_preds = best_model.predict(X_train)
    val_preds = best_model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(target_train, train_preds))
    train_mape = mean_absolute_percentage_error(target_train, train_preds) * 100
    
    val_rmse = np.sqrt(mean_squared_error(target_val, val_preds))  
    val_mape = mean_absolute_percentage_error(target_val, val_preds) * 100 

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(target_train.index, target_train, label='Actual Train', color='blue', alpha=0.7)
        plt.plot(target_val.index, target_val, label='Actual Validation', color='green', alpha=0.7)
        plt.plot(target_val.index, val_preds, label='Predicted Validation', color='red')
        plt.legend()
        plt.title(f'{best_model_name.upper()} Model - Training and Validation Predictions')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.show()

    results = {
        "model_type": best_model_name,
        "features_used": X_train.columns.tolist(),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mape (%)": train_mape,
        "val_mape (%)": val_mape
        }

    if csv_path:
        results_df = pd.DataFrame([results])
        
        # Check if file exists to determine mode
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, mode='w', header=True, index=False)
        
        print(f"Results appended to {csv_path}")

    return best_model, val_preds

# XGBoost

def xgboost_regressor(X_train, X_val, target_train, target_val, plot=False, csv_path=None):
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
    
    # Fit the model
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
        plt.title(f'XGBoost Model - Training and Validation Predictions')
        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.show()

    # Store results
    results = {
        "model_type": "XGBoost",
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

# Using ChatGPT help to try to find the error

lazy_model = {}
lazy_pred = {}

# Iterate over fs_mm_xgb and process each target variable
for target_name, selected_features in features.items():
    try:
        # Extract the corresponding target train and validation data
        target_number = target_name.split('_')[-1]
        
        # Instead of using globals, directly get target_train and target_val
        target_train = globals().get(f'y_train_{target_number}')
        target_val = globals().get(f'y_val_{target_number}')
        
        if target_train is None or target_val is None:
            print(f"Warning: Target data for {target_name} is missing.")
            continue  # Skip to the next target variable if data is missing
        
        # Extract the selected features
        X_train_target = X_train_scaled[selected_features]
        X_val_target = X_val_scaled[selected_features]
        
        # Check if the selected features are valid (no missing columns)
        missing_columns = [col for col in selected_features if col not in X_train_scaled.columns]
        if missing_columns:
            print(f"Warning: Missing columns in selected features for {target_name}: {missing_columns}")
            continue
        
        # Fit LazyRegressor and store results in the dictionaries
        print(f"Fitting LazyRegressor for target {target_name}...")
        lazy_model[target_number], lazy_pred[target_number] = fs.lazy_regressor(
            X_train_target, X_val_target, 
            target_train, target_val, 
            plot=False, 
            csv_path=f'./modelling_csvs/{target_number}_results.csv'
        )
        
        # Instead of checking 'empty', check if a model was trained
        if lazy_model[target_number] is None or isinstance(lazy_model[target_number], Pipeline) and not hasattr(lazy_model[target_number], 'steps'):
            print(f"Warning: No valid model trained for {target_name}.")
            continue

    except Exception as e:
        # Catch and report any errors during the process
        print(f"Error processing {target_name}: {e}")

# For some reason when processing #1 inside the loop we receive an error: attempt to get argmin of an empty sequence
# However, when using the same function for the same #1 outside a loop (bellow) everything works fine

feat = features['y_train_5']

lazy_model['5'], lazy_pred['5'] = fs.lazy_regressor(X_train_scaled[feat], X_val_scaled[feat], 
               y_train_4, 
               y_val_4, plot = True, 
               csv_path = './modelling_csvs/5_results.csv')