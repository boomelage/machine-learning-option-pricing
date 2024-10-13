import sys
import os
import time
from pathlib import Path
from model_settings import ms
import pandas as pd
import numpy as np
from model_settings import ms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import QuantLib as ql
from itertools import product

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = str(Path().resolve().parent)
os.chdir(current_dir)
sys.path.append(parent_dir)
from historical_av_key_collector import keys_df, symbol
storefile = os.path.join(parent_dir,f'alphaVantage {symbol}.h5')
keys_df = keys_df.copy()[['surface_key','contract_key','raw_data_key','calibration_key']].fillna(0)
keys_df = keys_df[
    (
    (keys_df['calibration_key']==0)
    )
]

print(keys_df.describe().iloc[0])

for i,row in keys_df.iterrows():
    surface_key = row['surface_key']
    contract_key = row['contract_key']
    raw_data_key = row['raw_data_key']
    
    second_backslash_pos = surface_key.find('/', 1)
    date_key_component = surface_key[:second_backslash_pos+1]
    date = surface_key[surface_key.find('_',0)+1:surface_key.find('_',0)+11]
    
    calculation_datetime = datetime.strptime(date,'%Y_%m_%d')
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year
    )
    printdate = calculation_datetime.strftime('%A, %Y-%m-%d')
    """
    HDF5 collection
    """
    while True:
        try:
            store = pd.HDFStore(os.path.join(parent_dir,storefile))
            raw_data = pd.DataFrame(store[raw_data_key])
            contracts = pd.DataFrame(store[contract_key])
            vol_matrix = pd.DataFrame(store[surface_key])
            break
        except OSError:
            print(OSError)
            print('retrying in...')
            for i in range (0,5):
                print(5-i)
                time.sleep(1)
        finally:
            store.close()
    
    vol_matrix = vol_matrix.sort_index().drop_duplicates()
    pd.to_numeric(raw_data['last'])
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data['expiration'] = pd.to_datetime(raw_data['expiration'])

    for vol in raw_data['implied_volatility']:
        if vol.find('-') != -1:
            print(vol)
    raw_data = raw_data[~raw_data['implied_volatility'].str.contains('-', na=False)]    


    raw_data['implied_volatility'] = pd.to_numeric(raw_data['implied_volatility']).astype(float)
    raw_data['strike'] = pd.to_numeric(raw_data['strike'])
    raw_data['last'] = pd.to_numeric(raw_data['last'])
    contract_maturities = np.array(
        (raw_data['expiration'] - raw_data['date']) / timedelta(days=1)).astype(int)
    raw_data['days_to_maturity'] = contract_maturities
    
    s = float(contracts['spot_price'].unique()[0])
    T = vol_matrix.columns.tolist()
    K = vol_matrix.index.tolist()
    r = 0.04
    g = 0.018
    
    ql.Settings.instance().evaluationDate = calculation_date
    flat_ts, dividend_ts = ms.ql_ts_rg(r, g, calculation_date)
    S_handle = ql.QuoteHandle(ql.SimpleQuote(s))
    
    heston_helpers = []
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; eta = 0.5;
    process = ql.HestonProcess(
        flat_ts,
        dividend_ts,
        S_handle,
        v0,                # Initial volatility
        kappa,             # Mean reversion speed
        theta,             # Long-run variance (volatility squared)
        eta,               # Volatility of the volatility
        rho                # Correlation between asset and volatility
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    
    for t in T:
        for k in K:
            p = ql.Period(int(t),ql.Days)
            volatility = vol_matrix.loc[k,t]
            helper = ql.HestonModelHelper(
                p, ms.calendar, float(s), k, 
                ql.QuoteHandle(ql.SimpleQuote(volatility)), 
                flat_ts, 
                dividend_ts
                )
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
    
    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    
    
    model.calibrate(heston_helpers, lm,
                      ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))
    
    theta, kappa, eta, rho, v0 = model.params()
    heston_parameters = pd.Series(
        [theta, kappa, eta, rho, v0],
        index = ['theta', 'kappa', 'eta', 'rho', 'v0'],
        dtype = float
    )
    
    calibration_data = raw_data.copy()[['strike','type','last','implied_volatility','days_to_maturity']]
    calibration_data.columns = ['strike_price','w','market_price','volatility','days_to_maturity']
    calibration_data['spot_price'] = s
    calibration_data['risk_free_rate'] = r
    calibration_data['dividend_rate'] = g
    calibration_data = calibration_data[calibration_data['days_to_maturity'].isin(contracts['days_to_maturity'])]
    calibration_data = calibration_data[calibration_data['days_to_maturity'].isin(contracts['days_to_maturity'])]
    
    calibration_data[heston_parameters.index.tolist()] = np.tile(heston_parameters,(calibration_data.shape[0],1))
    calibration_data.loc[:,'moneyness'] = ms.vmoneyness(calibration_data['spot_price'].values,calibration_data['strike_price'].values,calibration_data['w'].values)
    calibration_data['calculation_date'] = calculation_datetime.strftime('%Y-%m-%d')
    calibration_data['black_scholes'] = ms.vector_black_scholes(calibration_data)
    calibration_data['heston_price'] = ms.vector_heston_price(calibration_data)
    calibration_data.loc[:,'absolute_error'] = calibration_data['heston_price'].values/calibration_data['black_scholes'].values-1
    avg = np.mean(np.abs(calibration_data['absolute_error']))
    print(f"\n{printdate}\n{heston_parameters}\naverage absolute error: {round(avg,3)}")

    """
    HDF5 storage
    """
    while True:
        try:
            store = pd.HDFStore(os.path.join(parent_dir,storefile))
            store.put(
                    f"{date_key_component}calibration", 
                    calibration_data,
                    format='table',
                    # append=True
                )
            break
        except Exception as e:
            print(e)
            print('retrying in...')
            for i in range (2):
                print(2-i)
                time.sleep(1)
        finally:
            store.close()