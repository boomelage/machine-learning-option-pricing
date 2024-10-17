import time
import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import datetime
from datetime import timedelta
from model_settings import ms
from historical_av_database_indexing import h5_name, keys_df

keys = keys_df[
	(keys_df['calibration_results'].isna())
	&
	(~keys_df['surface'].isna())
]['raw_data'].tolist()

for key in keys:
	locator = key[:key.find('/',1)+1]
	surface_key = str(locator+'surface')
	spot_price_key = str(locator+'spot_price')
	date = key[key.find('_',0)+1:key.find('/',1)].replace('_','-')
	while True:
		try:
			with pd.HDFStore(h5_name) as store:
				raw_data = store[key]
				vol_matrix = store[surface_key]
				s = store[spot_price_key].iloc[0]
			break
		except Exception as e:
			print(e)
			time.sleep(2)
		finally:
			store.close()
	calculation_datetime = datetime.strptime(date,'%Y-%m-%d')
	calculation_date = ql.Date(
		calculation_datetime.day,
		calculation_datetime.month,
		calculation_datetime.year
	)
	printdate = calculation_datetime.strftime("%A, ") + str(calculation_date)
	print(f"\nattempting calibration for {date}")
	try:
		vol_matrix = vol_matrix.sort_index().drop_duplicates()
		raw_data['date'] = pd.to_datetime(raw_data['date'])
		raw_data['expiration'] = pd.to_datetime(raw_data['expiration'])
		raw_data['implied_volatility'] = pd.to_numeric(raw_data['implied_volatility']).astype(float)
		raw_data['strike'] = pd.to_numeric(raw_data['strike'])
		raw_data['last'] = pd.to_numeric(raw_data['last'])
		contract_maturities = np.array((raw_data['expiration'] - raw_data['date']) / timedelta(days=1)).astype(int)
		raw_data['days_to_maturity'] = contract_maturities

		T = vol_matrix.columns.tolist()
		K = vol_matrix.index.tolist()
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
		calibration_test_data = raw_data.copy()[['strike','type','last','implied_volatility','days_to_maturity']]
		calibration_test_data.columns = ['strike_price','w','market_price','volatility','days_to_maturity']
		calibration_test_data['calculation_date'] = date
		calibration_test_data['spot_price'] = s
		calibration_test_data['risk_free_rate'] = r
		calibration_test_data['dividend_rate'] = g
		calibration_test_data = calibration_test_data[calibration_test_data['strike_price']>0.5*s]
		calibration_test_data = calibration_test_data[calibration_test_data['days_to_maturity'].isin(T)]
		calibration_test_data[heston_parameters.index.tolist()] = np.tile(heston_parameters,(calibration_test_data.shape[0],1))
		calibration_test_data.loc[:,'moneyness'] = ms.vmoneyness(
		    calibration_test_data['spot_price'].values,
		    calibration_test_data['strike_price'].values,
		    calibration_test_data['w'].values)
		calibration_test_data['calculation_date'] = str(date)
		calibration_test_data['black_scholes'] = ms.vector_black_scholes(calibration_test_data)
		calibration_test_data['heston_price'] = ms.vector_heston_price(calibration_test_data)

		calibration_test_data.loc[:,'error'] = calibration_test_data['heston_price'].values/calibration_test_data['black_scholes'].values - 1
		avg = np.mean(np.abs(calibration_test_data['error']))
		print(f"\n{printdate}\n{heston_parameters}\naverage absolute relative error: {round(avg*100,4)}%")
		date_key_component = 'date_' + date.replace('-','_') + '/'
		while True:
			try:
				with pd.HDFStore(h5_name) as store:
					store.put(
						f"{date_key_component}heston_calibration/heston_parameters",
						heston_parameters,
						format='fixed',
						append=False
					)
					store.put(
						f"{date_key_component}heston_calibration/calibration_results",
						calibration_test_data,
						format='table',
						append=False
					)
				print(f"data stored for {printdate}")
				break
			except Exception as e:
				print(e)
				time.sleep(2)
			finally:
				
				store.close()
	except Exception as e:
		print(f"calibration aborted:")
		raise(e)
