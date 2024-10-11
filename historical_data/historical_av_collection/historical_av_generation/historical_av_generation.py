import time
import sys
import os 
import pandas as pd
import modin.pandas as md
import QuantLib as ql
import numpy as np
from tqdm import tqdm
from datetime import datetime
from itertools import product
from pathlib import Path
from model_settings import ms


def generate_barrier_features(s, K, T, barriers, updown, OUTIN, W):
    barrier_features = pd.DataFrame(
        product([s], K, barriers, T, [updown], OUTIN, W),
        columns=[
            'spot_price', 'strike_price', 'barrier', 'days_to_maturity',
            'updown', 'outin', 'w'
        ]
    )
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features



current_dir = str(Path().resolve())
parent_dir = str(Path().resolve().parent)

while True:
	try:

		store = pd.HDFStore(os.path.join(parent_dir,'alphaVantage vanillas.h5'))
		keys = store.keys()

		contracts_keys = pd.Series([key for key in keys if key.find('hottest_contracts')!=-1])
		raw_data_keys = pd.Series([key for key in keys if key.find('raw_data')!=-1])
		surface_keys = pd.Series([key for key in keys if key.find('surface')!=-1])
		calibrations_keys = pd.Series([key for key in keys if key.find('calibration_test')!=-1])
		priced_secturities_keys = pd.Series([key for key in keys if key.find('priced_securities')!=-1])

		break
	except OSError:
		print(OSError)
		print('retrying in...')
		for i in range(2):
			print(2-i)
			time.sleep(1)
	finally:
		store.close()

keys_df = pd.DataFrame(
	{
	'contracts_key':contracts_keys,
	'raw_data_key':raw_data_keys,
	'surface_key':surface_keys,
	'calibration_key':calibrations_keys,
	'priced_securities_key':priced_secturities_keys
	}
).fillna(0)
print(f'\n{keys_df}')
keys_df = keys_df[
	(
		(keys_df['calibration_key']!=0)
		# &
		# (keys_df['priced_securities_key']==0)
	)
]
print(f'\n{keys_df}')

bar = tqdm(total = keys_df.shape[0])

for i,row in keys_df.iterrows():
	contracts_key = row['contracts_key']
	raw_data_key = row['raw_data_key']
	surface_key = row['surface_key']
	calibration_key = row['calibration_key']


	while True:
		try:
			store = pd.HDFStore(os.path.join(parent_dir,'alphaVantage vanillas.h5'))
			hottest_contracts = store[contracts_key]
			raw_data = store[raw_data_key]
			surface = store[surface_key]
			calibration = store[calibration_key]
			break
		except OSError:
			print(OSError)
			print('retrying in...')
			for i in range(2):
				print(2-i)
				time.sleep(1)
		finally:
			store.close()

	heston_parameters = calibration[['theta','kappa','rho','eta','v0']].drop_duplicates().squeeze()

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

	s = float(hottest_contracts['spot_price'].unique()[0])
	r = 0.04
	g = 0.018

	ql.Settings.instance().evaluationDate = calculation_date
	flat_ts, dividend_ts = ms.ql_ts_rg(r, g, calculation_date)
	S_handle = ql.QuoteHandle(ql.SimpleQuote(s))


	rebate = 0.
	step = 1
	atm_spread = 1
	r = 0.04
	K = np.linspace(
	    s*0.9,
	    s*1.1,
	   	50
	    )
	T = [
	    60,
	    90,
	    180,
	    360,
	    540,
	    720
	    ]
	OUTIN = ['Out','In']
	W = ['call','put']
	    

	barriers = np.linspace(
	    s*0.5,s*0.99,
	    5
	    ).astype(float).tolist()
	down_features = generate_barrier_features(
	    s, K, T, barriers, 'Down', OUTIN, W)


	barriers = np.linspace(
	    s*1.01,s*1.5,
	    5
	    ).astype(float).tolist()
	up_features = generate_barrier_features(
	    s, K, T, barriers, 'Up', OUTIN, W)


	features = pd.concat(
	    [down_features,up_features],
	    ignore_index = True
	    )
	features['rebate'] = rebate
	features['dividend_rate'] = g
	features['risk_free_rate'] = r
	features.loc[:,'theta'] = heston_parameters['theta']
	features.loc[:,'kappa'] = heston_parameters['kappa']
	features.loc[:,'rho'] = heston_parameters['rho']
	features.loc[:,'eta'] = heston_parameters['eta']
	features.loc[:,'v0'] = heston_parameters['v0']
	features['calculation_date'] = calculation_datetime
	features['expiration_date'] =  calculation_datetime + pd.to_timedelta(
			features['days_to_maturity'], unit='D')

	features['heston_vanilla'] = ms.vector_heston_price(features)
	features['barrier_price'] = ms.vector_barrier_price(features)

	while True:
		try:
			store = pd.HDFStore(os.path.join(parent_dir,'alphaVantage vanillas.h5'))
			store.append(f"{date_key_component}priced_securities", features, format='table',append=True)
			print(f'\ndata stored for {printdate}')
			break
		except OSError:
			print(f"\nerror for {printdate}:\nOSError")
			print('retrying in...')
			for i in range(2):
				print(2-i)
				time.sleep(1)
		finally:
			store.close()
	bar.update(1)
bar.close()
