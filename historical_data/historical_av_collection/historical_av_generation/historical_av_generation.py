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
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = str(Path().resolve().parent)
os.chdir(current_dir)
sys.path.append(parent_dir)
from historical_av_underlying_fetcher import symbol

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


from historical_av_key_collector import keys_df

# keys_df = keys_df.dropna(subset='calibration_key').fillna(0)
# keys_df = keys_df.copy()[keys_df['priced_securities_key']==0]
# print(f"\n{keys_df[['contract_key','priced_securities_key']]}")

bar = tqdm(total = keys_df.shape[0])

for i,row in keys_df.iterrows():
	contract_key = row['contract_key']
	raw_data_key = row['raw_data_key']
	surface_key = row['surface_key']
	calibration_key = row['calibration_key']


	while True:
		try:
			store = pd.HDFStore(os.path.join(parent_dir,f'alphaVantage {symbol}.h5'))
			hottest_contracts = store[contract_key]
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
	features['theta'] = heston_parameters['theta']
	features['kappa'] = heston_parameters['kappa']
	features['rho'] = heston_parameters['rho']
	features['eta'] = heston_parameters['eta']
	features['v0'] = heston_parameters['v0']
	features['calculation_date'] = calculation_datetime.strftime('%Y-%m-%d')
	features['expiration_date'] =  (calculation_datetime + pd.to_timedelta(
			features['days_to_maturity'], unit='D'))
	features['expiration_date'] = features['expiration_date'].dt.strftime('%Y-%m-%d')

	features['heston_vanilla'] = ms.vector_heston_price(features)
	features['barrier_price'] = ms.vector_barrier_price(features)

	while True:
		try:
			store = pd.HDFStore(os.path.join(parent_dir,f'alphaVantage {symbol}.h5'))
			store.put(
				f"{date_key_component}priced_securities", 
				features, 
				format='table',
				# append=True
				)
			print(f'\ndata stored for {printdate}')
			break
		except Exception as e:
			print(f"\nerror for {printdate}:\n{e}")
			print('retrying in...')
			for i in range(2):
				print(2-i)
				time.sleep(1)
		finally:
			store.close()
	bar.update(1)
bar.close()