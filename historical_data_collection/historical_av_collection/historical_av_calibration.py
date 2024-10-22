import os
import time
import requests
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
from model_settings import vanilla_pricer, ms
from heston_model_calibration import calibrate_heston
from historical_av_options_collector import collect_av_link
vanp = vanilla_pricer()

symbol = 'SPY'
h5_name = f"alphaVantage {symbol}.h5"

url = str(
    'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+
    symbol+'&outputsize=full&apikey='+
    ms.av_key)
print(symbol)
r = requests.get(url)
spots = pd.Series(pd.DataFrame(r.json()['Time Series (Daily)']).transpose()['4. close'].squeeze())
spots = pd.to_numeric(spots,errors='coerce')
print(spots)


for date,s in spots.items():
    raw_data = collect_av_link(date,symbol)
    calculation_datetime = datetime.strptime(str(date),'%Y-%m-%d')
    raw_data['spot_price'] = s
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data['expiration'] = pd.to_datetime(raw_data['expiration'])
    raw_data['implied_volatility'] = pd.to_numeric(raw_data['implied_volatility'],errors='coerce').astype(float)
    contract_maturities = np.array((raw_data['expiration'] - raw_data['date']) / timedelta(days=1)).astype(int)
    raw_data['days_to_maturity'] = contract_maturities
    float_columns = [
        'strike', 'last', 'mark','bid', 'ask', 'implied_volatility', 
        'delta', 'gamma', 'theta', 'vega', 'rho'
    ]
    int_columns = ['bid_size','ask_size', 'volume', 'open_interest']
    raw_data[int_columns] = raw_data[int_columns].astype(int)
    raw_data[float_columns] = raw_data[float_columns].astype(float)
    raw_data = raw_data[raw_data['days_to_maturity']>0].reset_index(drop=True)
    raw_data['moneyness'] = ms.vmoneyness(raw_data['spot_price'],raw_data['strike'],raw_data['type'])
    T = np.sort(raw_data['days_to_maturity'].unique())
    
    calls = raw_data[raw_data['type']=='call'].copy()

    calls['moneyness']=np.abs(calls['moneyness'])
    calls = calls.sort_values(by='moneyness',ascending=True)
    by_t = calls.groupby('days_to_maturity')
    atm_vols = {}
    for t in T:
        atm_vols[t] = by_t.get_group(t).iloc[0].loc['implied_volatility']
    atm_vols = pd.Series(atm_vols)

    raw_data = raw_data[
        (raw_data['moneyness']<0)
        &(raw_data['moneyness']>=-0.05)
    ]


    T = np.sort(raw_data['days_to_maturity'].unique())
    K = np.sort(raw_data['strike'].unique())
    
    by_kt = raw_data[raw_data['days_to_maturity'].isin(T)].copy().set_index(['strike','days_to_maturity'])
    
    raw_surf = pd.DataFrame(np.tile(np.nan,(len(K),len(T))),index=K,columns=T)
    for k in K:
        for t in T:
            try:
                raw_surf.loc[k,t] = by_kt.loc[(k,t),'implied_volatility']
            except Exception:
                pass

    market_surface = raw_surf.dropna().copy()
    market_surface = pd.DataFrame(np.tile(np.nan,(len(K),len(T))),index=K,columns=T)
    for k in K:
        for t in T:
            try:
                market_surface.loc[k,t] = by_kt.loc[(k,int(t)),'implied_volatility']
            except Exception:
                pass
    market_surface = market_surface.dropna()

    T1 = int(np.max(T[T<=7]))
    T2 = int(np.max(T[(T>T1)&(T<=31)]))
    T3 = int(np.max(T[(T>T2)&(T<=100)]))
    T4 = int(np.max(T[(T>T3)&(T<=200)]))
    T5 = int(np.max(T[(T>T4)&(T<=400)]))
    T6 = int(np.max(T[(T>T5)&(T<=800)]))
    T = np.array([T1,T2,T3,T4,T5,T6],dtype=int)
    indices = np.linspace(0, len(raw_surf.index) - 1, num=5, dtype=int)
    K = raw_surf.index[indices]
    
    market_surface = market_surface.loc[:,T]

    r = 0.04
    g = 0.0
    try:
        heston_parameters = calibrate_heston(market_surface,s,r,g)
        print(heston_parameters,calculation_datetime)


        test_data = pd.DataFrame(
            product(
                [s],
                K,
                T,
            ),
            columns = [
                'spot_price','strike_price','days_to_maturity'
            ]
        )
        test_data = by_kt.copy().reset_index()
        test_data = test_data[
            (test_data['strike'].isin(K))&
            (test_data['days_to_maturity'].isin(T))
        ]
        test_data['risk_free_rate'] = r
        test_data['dividend_rate'] = 0.00
        test_data = test_data.rename(columns={'strike':'strike_price','implied_volatility':'volatility','type':'w','date':'calculation_date'})
        for param, value in heston_parameters.items():
            test_data[param] = value
        test_data['black_scholes'] = vanp.df_numpy_black_scholes(test_data)
        test_data['heston'] = vanp.df_heston_price(test_data)
        test_data['relative_error'] = test_data['heston']/test_data['black_scholes']-1
        calibration_error = np.mean(np.abs(test_data['relative_error']))
        print(test_data.iloc[:,-3:], calibration_error)
        test_data.to_csv(os.path.join('av_calibrations',f'{symbol} calbirated {calculation_datetime.strftime('%Y-%m-%d')}.csv'))

    except Exception as e:
        print(e)
        pass