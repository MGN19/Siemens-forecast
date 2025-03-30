rename_dict = {
    'China': 'CHI Production Index',
    'China.1': 'CHI Shipments Index',
    'France': 'FRA Production Index',
    'France.1': 'FRA Shipments Index',
    'Germany': 'GER Production Index',
    'Germany.1': 'GER Shipments Index',
    'Italy': 'ITA Production Index',
    'Italy.1': 'ITA Shipments Index',
    'Japan': 'JAP Production Index',
    'Japan.1': 'JAP Shipments Index',
    'Switzerland': 'SWI Production Index',
    'Switzerland.1': 'SWI Shipments Index',
    'United Kingdom': 'UK Production Index',
    'United Kingdom.1': 'UK Shipments Index',
    'United States': 'USA Production Index',
    'United States.1': 'USA Shipments Index',
    'Europe': 'Europe Production Index',
    'Europe.1': 'Europe Shipments Index',
    'Unnamed: 19': '(W) Price of Base Metals',
    'Unnamed: 20': '(W) Price of Energy',
    'Unnamed: 21': '(W) Price of Metals & Minerals',
    'Unnamed: 22': '(W) Price of Natural gas index',
    'Unnamed: 23': '(W) Price of Crude oil, average',
    'Unnamed: 24': '(W) Price of Copper',
    'Unnamed: 25': 'USA EUR to LCU Conversion Rate ',
    'Producer Prices': 'USA EE Producer Prices',
    'Producer Prices.1': 'UK EE Producer Prices',
    'Producer Prices.2': 'ITA EE Producer Prices',
    'Producer Prices.3': 'FRA EE Producer Prices',
    'Producer Prices.4': 'GER EE Producer Prices',
    'Producer Prices.5': 'CHI EE Producer Prices',
    'production index': 'USA Machinery & Equipment Index',
    'production index.1': '(W) Machinery & Equipment Index',
    'production index.2': 'SWI Machinery & Equipment Index',
    'production index.3': 'UK Machinery & Equipment Index',
    'production index.4': 'ITA Machinery & Equipment Index',
    'production index.5': 'JAP Machinery & Equipment Index',
    'production index.6': 'FRA Machinery & Equipment Index',
    'production index.7': 'GER Machinery & Equipment Index',
    'production index.8': 'USA EE Production Index',
    'production index.9': '(W) EE Production Index',
    'production index.10': 'SWI EE Production Index',
    'production index.11': 'UK EE Production Index',
    'production index.12': 'ITA EE Production Index',
    'production index.13': 'JAP EE Production Index',
    'production index.14': 'FRA EE Production Index',
    'production index.15': 'GER EE Production Index'
}

rename_dict_stocks = {
    'Price': 'stock_price',
    'Change %': 'stock_price_change',
    'Volume': 'stock_volume',

}


rename_dict_consumer = {
    'China (People’s Republic of)': 'CC_CHI',
    'France': 'CC_FRA',
    'Germany': 'CC_GER',
    'Italy': 'CC_ITA',
    'Japan': 'CC_JAP',
    'OECD Europe': 'CC_Europe',
    'Switzerland': 'CC_SWI',
    'United Kingdom': 'CC_UK',
    'United States': 'CC_USA'
}

rename_dict_business = {
    'China (People’s Republic of)': 'BC_CHI',
    'France': 'BC_FRA',
    'Germany': 'BC_GER',
    'Italy': 'BC_ITA',
    'Japan': 'BC_JAP',
    'OECD Europe': 'BC_Europe',
    'Switzerland': 'BC_SWI',
    'United Kingdom': 'BC_UK',
    'United States': 'BC_USA'
}

rename_dict_covid = {
    'China': 'Covid_Chi',
    'France': 'Covid_Fra',
    'Germany': 'Covid_Ger',
    'Italy': 'Covid_Ita',
    'Switzerland': 'Covid_Swi',
    'United Kingdom': 'Covid_UK',
    'United States': 'Covid_US'
}

rename_dict_clean = {
    'China': 'Clean_Chi',
    'France': 'Clean_Fra',
    'Germany': 'Clean_Ger',
    'Italy': 'Clean_Ita',
    'Switzerland': 'Clean_Swi',
    'United Kingdom': 'Clean_UK',
    'United States of America': 'Clean_US'
}

rename_dict_fossil = {
    'China': 'Fossil_Chi',
    'France': 'Fossil_Fra',
    'Germany': 'Fossil_Ger',
    'Italy': 'Fossil_Ita',
    'Switzerland': 'Fossil_Swi',
    'United Kingdom': 'Fossil_UK',
    'United States of America': 'Fossil_US'
}

rename_dict_buildings = {
    'Buildings Number': 'Buildings_Ger'
}

lag_X_dict = {
    # Economic & Market Indices
    "CHI Production Index": [12],
    "CHI Shipments Index": [12],
    "FRA Production Index": [12],
    "FRA Shipments Index": [12],
    "GER Production Index": [12],
    "GER Shipments Index": [12],
    "ITA Production Index": [12],
    "ITA Shipments Index": [12],
    "JAP Production Index": [12],
    "JAP Shipments Index": [12],
    "SWI Production Index": [12],
    "SWI Shipments Index": [12],
    "UK Production Index": [12],
    "UK Shipments Index": [12],
    "USA Production Index": [12],
    "USA Shipments Index": [12],
    "Europe Production Index": [12],
    "Europe Shipments Index": [12],

    # Prices of Commodities & Energy
    "(W) Price of Base Metals": [12],
    "(W) Price of Energy": [12],
    "(W) Price of Metals & Minerals": [12],
    "(W) Price of Natural gas index": [12],
    "(W) Price of Crude oil, average": [12],
    "(W) Price of Copper": [12],

    # Currency & Producer Prices
    "USA EUR to LCU Conversion Rate ": [12],
    "USA EE Producer Prices": [12],
    "UK EE Producer Prices": [12],
    "ITA EE Producer Prices": [12],
    "FRA EE Producer Prices": [12],
    "GER EE Producer Prices": [12],
    "CHI EE Producer Prices": [12],

    # Machinery & Equipment Indices
    "USA Machinery & Equipment Index": [12],
    "(W) Machinery & Equipment Index": [12],
    "SWI Machinery & Equipment Index": [12],
    "UK Machinery & Equipment Index": [12],
    "ITA Machinery & Equipment Index": [12],
    "JAP Machinery & Equipment Index": [12],
    "FRA Machinery & Equipment Index": [12],
    "GER Machinery & Equipment Index": [12],

    # Energy & Electricity Production Indices
    "USA EE Production Index": [12],
    "(W) EE Production Index": [12],
    "SWI EE Production Index": [12],
    "UK EE Production Index": [12],
    "ITA EE Production Index": [12],
    "JAP EE Production Index": [12],
    "FRA EE Production Index": [12],
    "GER EE Production Index": [12],

    # Stock Data
    "stock_price": [12],
    "stock_price_change": [12],
    "stock_volume": [12],

    # COVID-related Data
    "Covid_Chi": [12],
    "Covid_Fra": [12],
    "Covid_Ger": [12],
    "Covid_Ita": [12],
    "Covid_Swi": [12],
    "Covid_UK": [12],
    "Covid_US": [12],

    # Clean & Fossil Energy Data
    "Clean_Chi": [12],
    "Clean_Fra": [12],
    "Clean_Ger": [12],
    "Clean_Ita": [12],
    "Clean_Swi": [12],
    "Clean_UK": [12],
    "Clean_US": [12],
    "Fossil_Chi": [12],
    "Fossil_Fra": [12],
    "Fossil_Ger": [12],
    "Fossil_Ita": [12],
    "Fossil_Swi": [12],
    "Fossil_UK": [12],
    "Fossil_US": [12],
    
    # Construction Costs Data
    "CC_CHI": [12],
    "CC_FRA": [12],
    "CC_GER": [12],
    "CC_ITA": [12],
    "CC_JAP": [12],
    "CC_Europe": [12],
    "CC_SWI": [12],
    "CC_UK": [12],
    "CC_USA": [12],

    # Building Costs Data
    "BC_CHI": [12],
    "BC_FRA": [12],
    "BC_GER": [12],
    "BC_ITA": [12],
    "BC_JAP": [12],
    "BC_Europe": [12],
    "BC_SWI": [12],
    "BC_UK": [12],
    "BC_USA": [12],

    # Buildings Data
    "Buildings_Ger": [12]
}

# Depends on the ACF and PACF
lag_dict = {
    "#1": [1],
    "#3": [1],
    "#4": [1, 6],
    "#5": [1],
    "#6": [1],
    "#8": [1, 3],
    "#9": [1, 12],
    "#11": [1],
    "#12": [1, 3, 12],
    "#13": [1, 3],
    "#14": [1, 6],
    "#16": [1, 3],
    "#20": [1],
    "#36": [1,12]
}


# select features minmax XGBoost
fs_mm_xgb = {'y_train_36': ['#14_Lag_1', '#9_Lag_1', '#4_Lag_1'],
 'y_train_8': ['#36_Lag_1',
  'stock_price_change',
  'CHI Production Index',
  'SWI Production Index',
  '#36_Lag_12'],
 'y_train_20': ['#36_RollingMean_12', '#12_Lag_1', 'BC_CHI'],
 'y_train_9': ['#20_Lag_1', '#36_Lag_1', '#4_Lag_1', '#13_Lag_1'],
 'y_train_4': ['#12_Lag_12',
  'Buildings_Ger',
  'GER Production Index',
  '#12_RollingMean_3',
  '#3_Lag_1',
  '#1_Lag_1',
  'FRA Production Index',
  'BC_GER',
  '(W) Price of Base Metals',
  '#4_RollingMean_6',
  '#16_RollingMean_3'],
 'y_train_11': ['#3_Lag_1',
  '#14_Lag_1',
  'UK EE Producer Prices',
  'Fossil_Ita'],
 'y_train_5': ['#1_Lag_1',
  'Fossil_Swi',
  '#20_Lag_1',
  'Fossil_US',
  'TotalDaysInMonth',
  'Fossil_UK'],
 'y_train_12': ['#9_Lag_1',
  '#12_RollingMean_3',
  '#6_Lag_1',
  '#9_RollingMean_12',
  'Clean_Fra',
  '#20_Lag_1'],
 'y_train_13': ['#8_Lag_3', 'Clean_UK', '#13_RollingMean_3', 'Clean_US'],
 'y_train_6': ['Clean_Ita',
  'Fossil_Fra',
  '#20_Lag_1',
  '#13_RollingMean_3',
  '#8_Lag_3',
  'GER EE Production Index'],
 'y_train_16': ['GER Production Index',
  '#4_Lag_6',
  '#12_Lag_1',
  'SWI Production Index',
  '#16_RollingMean_3'],
 'y_train_3': ['#12_Lag_12', 'Buildings_Ger', 'Fossil_Chi', 'Clean_Fra'],
 'y_train_1': ['Clean_Fra', 'GER Production Index', 'Clean_US'],
 'y_train_14': ['#13_Lag_3', 'Clean_Ita', '#14_Lag_1']}

# select features robust XGBoost
fs_r_xgb ={'y_train_36': ['(W) Price of Energy', '#14_Lag_1', '#9_Lag_1', '#4_Lag_1'],
 'y_train_8': ['#36_Lag_1',
  '#36_Lag_12',
  'CHI Production Index',
  'UK EE Producer Prices',
  '#16_Lag_3'],
 'y_train_20': ['#36_RollingMean_12', 'BC_CHI'],
 'y_train_9': ['#36_Lag_1', '#4_Lag_1', '#13_Lag_1'],
 'y_train_4': ['#12_Lag_12',
  'GER Production Index',
  '#12_RollingMean_3',
  '#3_Lag_1',
  '#1_Lag_1',
  'BC_GER',
  'Month',
  '(W) Price of Base Metals',
  '#4_RollingMean_6',
  '#16_RollingMean_3'],
 'y_train_11': ['#3_Lag_1', '#14_Lag_1', 'UK EE Producer Prices'],
 'y_train_5': ['#1_Lag_1',
  'Fossil_Swi',
  '#20_Lag_1',
  'Fossil_US',
  '#5_Lag_1',
  'Fossil_UK'],
 'y_train_12': ['#9_Lag_1',
  '#12_RollingMean_3',
  '#9_RollingMean_12',
  'Clean_Fra',
  '#20_Lag_1'],
 'y_train_13': ['#13_RollingMean_3', 'Clean_US'],
 'y_train_6': ['Fossil_Fra',
  '#20_Lag_1',
  'JAP Production Index',
  '#13_RollingMean_3',
  '#8_Lag_3',
  'GER EE Production Index'],
 'y_train_16': ['SWI Production Index',
  '#12_Lag_1',
  '#16_RollingMean_3',
  '#4_Lag_6'],
 'y_train_3': ['#12_Lag_12',
  'Buildings_Ger',
  'Clean_Fra',
  'Clean_US',
  'Fossil_Chi',
  '#36_Lag_12'],
 'y_train_1': ['GER Production Index',
  'Clean_Fra',
  'Clean_Swi',
  'Month',
  'Clean_US'],
 'y_train_14': ['#13_Lag_3',
  'Clean_Ita',
  'Buildings_Ger',
  '#14_Lag_1',
  'stock_volume',
  'USA Production Index']}

# select features robust LGBM
fs_r_lgbm ={'y_train_36': ['#14_Lag_1', '#16_Lag_3'],
 'y_train_8': ['#16_Lag_3',
  '#12_RollingMean_12',
  '#12_Lag_3',
  '#36_Lag_1',
  '#36_Lag_12'],
 'y_train_20': ['#9_RollingMean_12',
  '#36_RollingMean_12',
  '#12_RollingMean_12'],
 'y_train_9': ['#14_Lag_6', '#36_Lag_1', '#13_Lag_1'],
 'y_train_4': ['#16_Lag_1',
  '#13_Lag_3',
  '#12_Lag_12',
  '#12_RollingMean_3',
  '#4_RollingMean_6',
  '#16_RollingMean_3'],
 'y_train_11': ['#9_RollingMean_12'],
 'y_train_5': ['#20_Lag_1', '#12_Lag_3'],
 'y_train_12': ['#12_Lag_12',
  '#12_Lag_3',
  '#12_RollingMean_3',
  '#9_RollingMean_12',
  '#14_Lag_1',
  '#36_Lag_12',
  '#20_Lag_1',
  '#16_Lag_3'],
 'y_train_13': ['#13_RollingMean_3'],
 'y_train_6': ['#20_Lag_1', '#16_Lag_1', '#14_Lag_6', '#12_Lag_12'],
 'y_train_16': ['#13_Lag_3',
  '#16_Lag_3',
  '#12_RollingMean_3',
  '#16_RollingMean_3'],
 'y_train_3': ['#20_Lag_1', '#12_Lag_12', '#36_Lag_12', '#13_Lag_1'],
 'y_train_1': [],
 'y_train_14': ['#13_Lag_3', '#14_Lag_1', '#13_RollingMean_3', '#12_Lag_3']}

fs_mm_lgbm2 = {'y_train_36': ['FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_8': ['SWI EE Production Index_Lag_12_RollingMean6', 'FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_20': ['UK EE Production Index_Lag_12_RollingMean6', 'UK EE Producer Prices_Lag_12_RollingMean3'], 'y_train_9': [], 'y_train_4': [], 'y_train_11': [], 'y_train_5': ['UK EE Production Index_Lag_12_RollingMean6'], 'y_train_12': ['FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_13': [], 'y_train_6': ['SWI EE Production Index_Lag_12_RollingMean3', 'USA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_16': ['FRA EE Producer Prices_Lag_12_RollingMean3', 'USA EE Producer Prices_Lag_12_RollingMean3', 'FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_3': [], 'y_train_1': [], 'y_train_14': ['USA EE Producer Prices_Lag_12_RollingMean3']}

fs_mm_lgbm3 = {'y_train_36': [], 'y_train_8': ['SWI EE Production Index_Lag_12_RollingMean6', '(W) EE Production Index_Lag_12_RollingMean6'], 'y_train_20': [], 'y_train_9': [], 'y_train_4': [], 'y_train_11': ['FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_5': ['UK EE Production Index_Lag_12_RollingMean6'], 'y_train_12': ['USA Shipments Index_Lag_12_RollingMean3'], 'y_train_13': ['USA Shipments Index_Lag_12_RollingMean3'], 'y_train_6': ['USA Shipments Index_Lag_12_RollingMean3', 'SWI EE Production Index_Lag_12_RollingMean3'], 'y_train_16': ['USA Shipments Index_Lag_12_RollingMean3', 'SWI EE Production Index_Lag_12_RollingMean3', 'FRA EE Producer Prices_Lag_12_RollingMean3'], 'y_train_3': ['USA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_1': ['USA Shipments Index_Lag_12_RollingMean3'], 'y_train_14': ['USA Shipments Index_Lag_12_RollingMean3', 'USA EE Producer Prices_Lag_12_RollingMean3', 'USA EE Producer Prices_Lag_12_RollingMean6']}

# min 100 features in RFE
fs_mm_xgb2 = {'y_train_36': ['CC_UK_Lag_12_RollingMean3', '#12_Lag_3', '#14_Lag_1', '#12_Lag_1_RollingMean6', '#9_Lag_12', '#1_Lag_1_RollingMean3', 'stock_volume_Lag_12', 'GER Production Index_Lag_12', '#36_Lag_12', '#8_Lag_3', '#16_Lag_1_RollingMean3', 'GER EE Production Index_Lag_12_RollingMean3', '#13_Lag_1'], 'y_train_8': ['CHI Production Index_Lag_12', 'USA EE Producer Prices_Lag_12_RollingMean3', 'FRA EE Producer Prices_Lag_12_RollingMean3', '#6_Lag_1_RollingMean3', '#13_Lag_3_RollingMean6', '#16_Lag_3', 'Clean_Ita_Lag_12', '#16_Lag_1_RollingMean3', '#11_Lag_1', '#12_Lag_12'], 'y_train_20': ['#16_Lag_3_RollingMean6', '#14_Lag_1_RollingMean6', '#6_Lag_1_RollingMean3', '#5_Lag_1', '#9_Lag_12', '#1_Lag_1_RollingMean6', '#16_Lag_1', 'Year', 'Fossil_US_Lag_12_RollingMean3', 'Fossil_Chi_Lag_12_RollingMean6', '#12_Lag_1_RollingMean3', '#3_Lag_1'], 'y_train_9': ['#1_Lag_1', 'CHI Production Index_Lag_12', '#20_Lag_1', 'CC_UK_Lag_12_RollingMean3', '#9_Lag_12_RollingMean3', '#14_Lag_1', '#36_Lag_1', 'GER EE Production Index_Lag_12', '#14_Lag_6', 'GER Production Index_Lag_12', 'Fossil_Chi_Lag_12_RollingMean6', '#4_Lag_1_RollingMean3'], 'y_train_4': ['#1_Lag_1', 'CHI Production Index_Lag_12', '#9_Lag_1_RollingMean3', 'Clean_Chi_Lag_12', '#9_Lag_1_RollingMean12', 'Clean_Fra_Lag_12', '#6_Lag_1_RollingMean3', '#3_Lag_1_RollingMean3', '#12_Lag_1', '#4_Lag_6_RollingMean6', 'SundayCount', 'Clean_Ita_Lag_12', 'Clean_US_Lag_12_RollingMean6', '#16_Lag_1', '#11_Lag_1_RollingMean6'], 'y_train_11': ['#1_Lag_1', 'BC_CHI_Lag_12', 'GerHolidayCount', '#11_Lag_1_RollingMean3', '#9_Lag_1_RollingMean3', '#9_Lag_12_RollingMean3', '#12_Lag_3', 'FRA EE Producer Prices_Lag_12_RollingMean6', '#14_Lag_1', '#3_Lag_1_RollingMean6', '#1_Lag_1_RollingMean3', '#4_Lag_6', '#12_Lag_1', '#8_Lag_1_RollingMean3', 'stock_price_Lag_12_RollingMean3', 'Fossil_Fra_Lag_12', '#3_Lag_1'], 'y_train_5': ['#1_Lag_1', '#20_Lag_1', '#5_Lag_1', 'TotalDaysInMonth', 'USA EUR to LCU Conversion Rate _Lag_12', '#4_Lag_6', '#16_Lag_3', 'stock_price_change_Lag_12', 'stock_volume_Lag_12', '#13_Lag_3_RollingMean3', 'CC_UK_Lag_12', '#4_Lag_1_RollingMean3'], 'y_train_12': ['CHI EE Producer Prices_Lag_12', 'CC_CHI_Lag_12', '#16_Lag_3', '#6_Lag_1_RollingMean6', 'Clean_Ita_Lag_12_RollingMean3', 'UK EE Producer Prices_Lag_12', '#6_Lag_1_RollingMean3', '#13_Lag_3_RollingMean6', '#6_Lag_1', '#9_Lag_1', '#16_Lag_1_RollingMean6', '#12_Lag_1_RollingMean3', 'Fossil_Fra_Lag_12', '#14_Lag_1_RollingMean3', '#12_Lag_3', '#14_Lag_1', '#36_Lag_1', 'USA EUR to LCU Conversion Rate _Lag_12', 'stock_price_change_Lag_12', '#12_Lag_1', '#9_Lag_1_RollingMean6', 'USA Shipments Index_Lag_12', '#4_Lag_1_RollingMean3', 'CHI Production Index_Lag_12', 'stock_price_Lag_12', '#20_Lag_1', '#36_Lag_12', '#9_Lag_1_RollingMean3'], 'y_train_13': ['USA Shipments Index_Lag_12_RollingMean3', 'Clean_US_Lag_12', '#6_Lag_1_RollingMean3', 'cos_month', '#6_Lag_1_RollingMean6', '#8_Lag_3', 'cos_Quarter', '#3_Lag_1'], 'y_train_6': ['CHI Production Index_Lag_12', '#20_Lag_1', '#11_Lag_1_RollingMean3', 'SWI EE Production Index_Lag_12_RollingMean3', '#20_Lag_1_RollingMean3', 'Clean_Chi_Lag_12', '#9_Lag_12_RollingMean3', 'USA EE Producer Prices_Lag_12', '#14_Lag_1_RollingMean6', '#5_Lag_1', '#12_Lag_3', '#14_Lag_6', '#6_Lag_1_RollingMean6', 'SWI EE Production Index_Lag_12', '#8_Lag_3', 'Fossil_Fra_Lag_12', '#3_Lag_1'], 'y_train_16': ['#3_Lag_1_RollingMean12', 'FRA EE Producer Prices_Lag_12_RollingMean3', '#9_Lag_12_RollingMean3', 'Fossil_US_Lag_12', 'USA EE Producer Prices_Lag_12', '#6_Lag_1_RollingMean3', 'Clean_Fra_Lag_12', 'JAP Production Index_Lag_12', '#9_Lag_1', '#12_Lag_1_RollingMean3', '#11_Lag_1_RollingMean3', '#12_Lag_3', '#4_Lag_6', '#9_Lag_1_RollingMean6', 'USA Shipments Index_Lag_12', '#14_Lag_6_RollingMean6', 'CC_UK_Lag_12_RollingMean3', '#9_Lag_12', '#4_Lag_6_RollingMean3', '#9_Lag_1_RollingMean3'], 'y_train_3': ['GerHolidayCount', '#20_Lag_1', '#14_Lag_1', 'SWI Production Index_Lag_12', 'TotalDaysInMonth', 'stock_price_change_Lag_12', '#36_Lag_12', '#8_Lag_3', '#12_Lag_12', 'Fossil_Chi_Lag_12_RollingMean3', 'USA Shipments Index_Lag_12'], 'y_train_1': ['CHI Production Index_Lag_12_RollingMean3', 'CC_UK_Lag_12_RollingMean3', '#14_Lag_1_RollingMean6', '#5_Lag_1', 'Clean_US_Lag_12', '#12_Lag_1_RollingMean6', '#9_Lag_12', '#4_Lag_6', '#3_Lag_1_RollingMean3', '#5_Lag_1_RollingMean3', 'Clean_UK_Lag_12', 'Clean_Ita_Lag_12_RollingMean3', 'sin_Quarter', 'ITA EE Producer Prices_Lag_12', 'GER EE Production Index_Lag_12_RollingMean3', 'Fossil_Fra_Lag_12'], 'y_train_14': ['USA Production Index_Lag_12', 'GerHolidayCount', 'USA Shipments Index_Lag_12_RollingMean3', '#14_Lag_1', '#9_Lag_1_RollingMean12', '#16_Lag_3', '#9_Lag_1', '#13_Lag_3_RollingMean3', '#36_Lag_12', '#13_Lag_1']}

# min 50 features in RFE
fs_mm_xgb3 = {'y_train_36': ['#9_Lag_12', '#8_Lag_3', '#1_Lag_1_RollingMean3', 'CC_UK_Lag_12_RollingMean3', '#12_Lag_1_RollingMean6', 'GER EE Production Index_Lag_12_RollingMean3', 'GER Production Index_Lag_12', 'FRA Production Index_Lag_12', '#36_Lag_12', '#14_Lag_1'], 'y_train_8': ['CHI Production Index_Lag_12', '#16_Lag_1_RollingMean3', 'Clean_Ita_Lag_12', '#11_Lag_1', '#12_Lag_12', '#11_Lag_1_RollingMean6', 'ITA EE Producer Prices_Lag_12'], 'y_train_20': ['#5_Lag_1', '#12_Lag_1_RollingMean3', '#16_Lag_3_RollingMean6', '#1_Lag_1_RollingMean6', '#3_Lag_1', 'Fossil_Chi_Lag_12_RollingMean6', '#16_Lag_1', 'Fossil_US_Lag_12_RollingMean3', 'Year'], 'y_train_9': ['GER EE Production Index_Lag_12', '#20_Lag_1', '#36_Lag_1', 'CC_UK_Lag_12_RollingMean3', 'GER Production Index_Lag_12', '#13_Lag_3_RollingMean6', '#9_Lag_12_RollingMean3', '#14_Lag_6'], 'y_train_4': ['CHI Production Index_Lag_12', 'SundayCount', '#6_Lag_1_RollingMean3', 'Clean_US_Lag_12_RollingMean6', '#4_Lag_6_RollingMean6', 'Clean_Chi_Lag_12', '#9_Lag_1_RollingMean3', 'Clean_Fra_Lag_12', '#1_Lag_1', '#11_Lag_1_RollingMean6', '#12_Lag_1'], 'y_train_11': ['#1_Lag_1_RollingMean3', 'WeekendDaysCount', '#8_Lag_1_RollingMean3', '#3_Lag_1', 'FRA EE Producer Prices_Lag_12_RollingMean6', 'GerHolidayCount', '#3_Lag_1_RollingMean6', '#9_Lag_1_RollingMean3', '#12_Lag_3', '#1_Lag_1', '#14_Lag_1', '#4_Lag_6', '#9_Lag_12_RollingMean3', 'ITA EE Producer Prices_Lag_12'], 'y_train_5': ['#5_Lag_1', 'TotalDaysInMonth', '#20_Lag_1', 'cos_month', '#16_Lag_3', 'CC_UK_Lag_12', '#4_Lag_1_RollingMean3', '#12_Lag_12', 'stock_volume_Lag_12', '#1_Lag_1', '#4_Lag_6'], 'y_train_12': ['#36_Lag_12_RollingMean3', 'Fossil_Fra_Lag_12', '#6_Lag_1_RollingMean6', '#13_Lag_3_RollingMean6', 'stock_price_Lag_12', '#6_Lag_1', 'CHI EE Producer Prices_Lag_12', '#16_Lag_3', '#9_Lag_1', 'ITA EE Producer Prices_Lag_12', '#14_Lag_1_RollingMean3', 'CC_CHI_Lag_12', '#36_Lag_1', '#9_Lag_1_RollingMean6', 'USA Shipments Index_Lag_12', '#20_Lag_1', '#4_Lag_1_RollingMean3', 'Clean_US_Lag_12', '#12_Lag_3', '#36_Lag_12', 'stock_price_change_Lag_12'], 'y_train_13': ['USA Shipments Index_Lag_12_RollingMean3', '#8_Lag_3', '#6_Lag_1_RollingMean3', '#13_Lag_3', 'cos_month', 'Clean_US_Lag_12', '#6_Lag_1_RollingMean6', 'Clean_Chi_Lag_12'], 'y_train_6': ['TotalDaysInMonth', '#14_Lag_1_RollingMean6', 'SWI EE Production Index_Lag_12', '#20_Lag_1', 'Fossil_Fra_Lag_12', '#3_Lag_1', '#6_Lag_1_RollingMean6', 'SWI Production Index_Lag_12_RollingMean3', '#11_Lag_1_RollingMean3', '#9_Lag_12_RollingMean3'], 'y_train_16': ['USA EUR to LCU Conversion Rate _Lag_12', '#6_Lag_1_RollingMean3', '#4_Lag_6_RollingMean3', 'USA Shipments Index_Lag_12', 'Fossil_US_Lag_12', 'JAP Production Index_Lag_12', 'USA EE Producer Prices_Lag_12', '#9_Lag_1_RollingMean3', '#11_Lag_1_RollingMean3', '#9_Lag_12_RollingMean3', 'stock_price_change_Lag_12'], 'y_train_3': ['TotalDaysInMonth', '#8_Lag_3', '#36_Lag_12', 'SWI Production Index_Lag_12', 'stock_price_change_Lag_12'], 'y_train_1': ['#5_Lag_1', '#9_Lag_12', 'Clean_Ita_Lag_12_RollingMean3', '#5_Lag_1_RollingMean3', 'CC_UK_Lag_12_RollingMean3', 'Fossil_Fra_Lag_12', 'Clean_US_Lag_12', 'Clean_Ita_Lag_12', '#3_Lag_1_RollingMean3', 'sin_Quarter', 'stock_price_Lag_12', 'CHI Production Index_Lag_12_RollingMean3'], 'y_train_14': ['USA Shipments Index_Lag_12_RollingMean3', '#9_Lag_1_RollingMean12', 'USA Production Index_Lag_12', '#13_Lag_3_RollingMean3', '#11_Lag_1_RollingMean12', '#16_Lag_3', 'GerHolidayCount', '#9_Lag_1', '#13_Lag_1', '#14_Lag_1', 'ITA EE Producer Prices_Lag_12']}

# min 20 features in RFE
fs_mm_xgb4 = {'y_train_36': ['#8_Lag_3', 'Year', '#9_Lag_12', 'CC_UK_Lag_12_RollingMean3', '#14_Lag_1'], 'y_train_8': ['#12_Lag_12', '#16_Lag_1_RollingMean3', 'CHI Production Index_Lag_12', '#11_Lag_1_RollingMean6', '#11_Lag_1', 'Clean_Ita_Lag_12'], 'y_train_20': ['Clean_Chi_Lag_12', '#16_Lag_1', '#12_Lag_1_RollingMean3', '#1_Lag_1_RollingMean6', '#5_Lag_1', 'Fossil_Chi_Lag_12_RollingMean6', '#16_Lag_3_RollingMean6', 'Fossil_US_Lag_12_RollingMean3'], 'y_train_9': ['#14_Lag_6', '#20_Lag_1', '#36_Lag_1', 'Fossil_Chi_Lag_12_RollingMean6', 'CC_UK_Lag_12_RollingMean3', 'GER EE Production Index_Lag_12'], 'y_train_4': ['#16_Lag_1', '(W) Price of Base Metals_Lag_12', '#6_Lag_1_RollingMean3', '#3_Lag_1', '#12_Lag_1', 'Clean_US_Lag_12_RollingMean6'], 'y_train_11': ['#12_Lag_3', '#1_Lag_1_RollingMean3', 'GerHolidayCount', '#3_Lag_1', 'FRA EE Producer Prices_Lag_12_RollingMean6'], 'y_train_5': ['#12_Lag_12', 'CC_UK_Lag_12', '#20_Lag_1', '#5_Lag_1', '#4_Lag_1_RollingMean3', '#1_Lag_1', 'stock_volume_Lag_12'], 'y_train_12': ['#9_Lag_1_RollingMean6', '#12_Lag_3', 'USA Shipments Index_Lag_12', '#14_Lag_1_RollingMean3', '#4_Lag_1_RollingMean3', '#9_Lag_1', '#6_Lag_1_RollingMean6', 'Clean_Fra_Lag_12', 'Clean_US_Lag_12'], 'y_train_13': ['#8_Lag_3', '#13_Lag_3', 'cos_month', '#6_Lag_1_RollingMean3', '#6_Lag_1_RollingMean6', 'Clean_US_Lag_12'], 'y_train_6': ['#11_Lag_1_RollingMean3', '#3_Lag_1', 'Fossil_Fra_Lag_12', 'USA EUR to LCU Conversion Rate _Lag_12', '#6_Lag_1_RollingMean6'], 'y_train_16': ['#4_Lag_6_RollingMean3', '#6_Lag_1_RollingMean3', 'USA Shipments Index_Lag_12', 'JAP Production Index_Lag_12', 'Fossil_US_Lag_12', '#9_Lag_1_RollingMean3', '#36_Lag_12'], 'y_train_3': ['#20_Lag_1', 'Year', 'TotalDaysInMonth', 'cos_month'], 'y_train_1': ['CHI Production Index_Lag_12_RollingMean3', 'Clean_Ita_Lag_12_RollingMean3', '#9_Lag_12', 'CC_UK_Lag_12_RollingMean3', 'Fossil_Fra_Lag_12', '#5_Lag_1_RollingMean3', 'stock_price_Lag_12', 'Clean_US_Lag_12'], 'y_train_14': ['USA Production Index_Lag_12', 'USA Shipments Index_Lag_12_RollingMean3', 'GerHolidayCount', '#11_Lag_1_RollingMean12', '#13_Lag_3_RollingMean3', '#14_Lag_1', '#9_Lag_1_RollingMean12']}