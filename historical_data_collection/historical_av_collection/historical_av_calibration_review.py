import time
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
from model_settings import vanilla_pricer, ms
from heston_model_calibration import calibrate_heston
vanp = vanilla_pricer()

symbol = 'SPY'
h5_name = f"alphaVantage {symbol}.h5"
with pd.HDFStore(h5_name) as store:
    keys = store.keys()
store.close()

cats = np.unique([k[k.rfind('/',0)+1:] for k in keys])
dates = np.unique([k[6:k.rfind('/',0)] for k in keys])
store_keys = {}
for date in dates:
    date_keys = [k for k in keys if k.find(date)!=-1]
    date_dict = {}
    for cat in cats:
        date_dict[cat] = [k for k in date_keys if k.find(cat)!=-1]
    store_keys[date] = date_dict

# dates = [dates[0]]
calibrations = []
for date in dates:
    keys = store_keys[date]
    with pd.HDFStore(h5_name) as store:
        try:
            raw_data = store[keys['raw_data'][0]]
            s = float(store[keys['spot_price'][0]].iloc[0])
        except Exception as e:
            print(keys)
    store.close()


    calculation_datetime = datetime.strptime(str(date),'%Y_%m_%d')
    # calculation_date = ql.Date(calculation_datetime.day,calculation_datetime.month,calculation_datetime.year)
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

    except Exception as e:
        print(e)
        pass