import pandas as pd
import numpy as np
import time
from model_settings import ms
from historical_av_key_collector import keys_df, symbol, h5_name
keys_df = keys_df.copy().dropna(subset=['spot_price','date'])
keys_df = keys_df[keys_df['surface_key'].isna()]

print(f"reconstructing {keys_df.shape[0]} surfaces")

for i,row in keys_df.iterrows():
	while True:
		try:
			with pd.HDFStore(h5_name) as store:
				raw_data = store[row['raw_data_key']]
				spot = store[row['spot_price']].iloc[0]
				date = store[row['date']].iloc[0]
			break
		except Exception as e:
			print(e)
			time.sleep(2)
		finally:
			store.close()


	df = raw_data.copy()
	columns_to_convert = ['strike', 'last', 'mark',
	       'bid', 'bid_size', 'ask', 'ask_size', 'volume', 'open_interest',
	       'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
	df[columns_to_convert] = df[
	    columns_to_convert].apply(pd.to_numeric, errors='coerce')

	df['expiration'] = pd.to_datetime(df['expiration'],format='%Y-%m-%d')
	df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
	df['days_to_maturity'] = df['expiration'] - df['date']
	df['days_to_maturity'] = df['days_to_maturity'] / np.timedelta64(1, 'D')
	df['days_to_maturity'] = df['days_to_maturity'].astype('int64')
	df = df[(df['days_to_maturity']>=30)&(df['days_to_maturity']<=400)]

	df = df[df['volume']>0].copy()
	df['spot_price'] = spot
	df['type'] = df['type'].str.lower()
	df['moneyness'] = ms.vmoneyness(df['spot_price'],df['strike'],df['type'])
	df = df[(df['moneyness']<0)&(df['moneyness']>-0.5)]
	indexed = df.copy().set_index(['strike','days_to_maturity'])

	T = np.sort(df['days_to_maturity'].unique()).tolist()
	K = np.sort(df['strike'].unique()).tolist()
	volume_heatmap = pd.DataFrame(
	    np.full((len(K), len(T)), np.nan), index=K, columns=T)
	for k in K:
	    for t in T:
	        try:
	            volume_heatmap.loc[k,t] = indexed.loc[(k,t),'volume']
	        except Exception:
	            pass
	        
	        
	hottest_contracts = pd.DataFrame(
	    volume_heatmap.unstack().sort_values(
	        ascending=False)).head(50).reset_index()
	hottest_contracts.columns = ['t','k','volume']
	T = np.sort(hottest_contracts['t'].unique()).tolist()
	K = np.sort(hottest_contracts['k'].unique()).tolist()

	vol_matrix = pd.DataFrame(
	    np.full((len(K),len(T)),np.nan),
	    index = K,
	    columns = T
	)
	for k in K:
	    for t in T:
	        try:
	            vol_matrix.loc[k,t] = indexed.loc[(k,float(t)),'implied_volatility']
	        except Exception:
	            pass

	vol_matrix = vol_matrix.dropna().copy()
	print(vol_matrix)
	print(date)
	while True:
		try:
			with pd.HDFStore(h5_name) as store:
				h5_key = f"date_{date.replace('-','_')}/surface"
				store.put(h5_key, vol_matrix, format='table', append=False)
				print(f"volatility surface stored for {date}")
			break
		except Exception as e:
			print(e)
			time.sleep(2)
		finally:
			store.close()