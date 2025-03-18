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