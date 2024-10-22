import time
import numpy as np
import pandas as pd
import QuantLib as ql
from Derman import derman
from datetime import datetime, timedelta
from model_settings import ms, vanilla_pricer
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

c = 0
calibrations = []
for date in dates:
    try:
        keys = store_keys[date]
        with pd.HDFStore(h5_name) as store:
            raw_data = store[keys['raw_data'][0]] 
            s = float(store[keys['spot_price'][0]].iloc[0])
        store.close()

        calculation_datetime = datetime.strptime(str(date),'%Y_%m_%d')
        calculation_date = ql.Date(calculation_datetime.day,calculation_datetime.month,calculation_datetime.year)
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
        raw_data = raw_data[
            (raw_data['moneyness']<0)
            &(raw_data['moneyness']>=-0.8)
        ]
        
        T = np.sort(raw_data['days_to_maturity'].unique())
        K = np.sort(raw_data['strike'].unique())
        
        T1 = int(np.max(T[T<=7]))
        T2 = int(np.max(T[(T>T1)&(T<=31)]))
        T3 = int(np.max(T[(T>T2)&(T<=90)]))
        T4 = int(np.max(T[(T>T3)&(T<=180)]))
        T5 = int(np.max(T[(T>T4)&(T<=370)]))
        T6 = int(np.max(T[(T>T5)&(T<=730)]))
        T = np.array([T1,T2,T3,T4,T5,T6],dtype=int)
        
        by_kt = raw_data[raw_data['days_to_maturity'].isin(T)].copy().set_index(['strike','days_to_maturity'])
        
        raw_surf = pd.DataFrame(np.tile(np.nan,(len(K),len(T))),index=K,columns=T)
        for k in K:
            for t in T:
                try:
                    raw_surf.loc[k,t] = by_kt.loc[(k,t),'implied_volatility']
                except Exception:
                    pass
        
        raw_surf = raw_surf.dropna()
        
        indices = np.linspace(0, len(raw_surf.index) - 1, num=5, dtype=int)
        
        K = raw_surf.index[indices]
        surface = raw_surf[raw_surf.index.isin(K)].copy()

        # surface = derman(surface,s)

        r = 0.04
        g = 0.0

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
                volatility = raw_surf.loc[k,t]
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
        calibration_test_data = raw_data.copy()[['strike','type','last','implied_volatility','days_to_maturity']]
        calibration_test_data.columns = ['strike_price','w','market_price','volatility','days_to_maturity']
        calibration_test_data['spot_price'] = s
        calibration_test_data['risk_free_rate'] = r
        calibration_test_data['dividend_rate'] = g
        calibration_test_data = calibration_test_data[calibration_test_data['days_to_maturity'].isin(T)]
        calibration_test_data = calibration_test_data[calibration_test_data['strike_price'].isin(K)]
        calibration_test_data[heston_parameters.index.tolist()] = np.tile(heston_parameters,(calibration_test_data.shape[0],1))
        calibration_test_data.loc[:,'moneyness'] = ms.vmoneyness(
            calibration_test_data['spot_price'].values,
            calibration_test_data['strike_price'].values,
            calibration_test_data['w'].values)
        calibration_test_data['calculation_date'] = calculation_datetime
        calibration_test_data['black_scholes'] = vanp.df_numpy_black_scholes(calibration_test_data)
        calibration_test_data['heston_price'] = vanp.df_heston_price(calibration_test_data)
        calibration_test_data.loc[:,'error'] = calibration_test_data['heston_price'].values/calibration_test_data['black_scholes'].values-1
        avg = np.mean(np.abs(calibration_test_data['error']))
        print(f"\n{heston_parameters}\naverage absolute relative error: {round(avg*100,3)}")
        if avg < 1:
            calibrations.append(calibration_test_data)


    except Exception as e:
        print(e)
        c += 1
        time.sleep(5)
        pass
print(c)

calibrations = pd.concat(calibrations,ignore_index=True)
print(calibrations)
calibrations.to_csv(r'alpha_vantage_new_calibration.csv')