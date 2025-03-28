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

# select features minmax XGBoost
{'y_train_36': ['#14_Lag_1', '#9_Lag_1', '#4_Lag_1'],
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