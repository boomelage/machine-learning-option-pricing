import pandas as pd
import numpy as np
from model_settings import ms
from historical_av_key_collector import keys_df, symbol, h5_name, available_dates
from historical_av_underlying_fetcher import historical_spots
keys_df = keys_df.dropna(subset=['raw_data_key','spot_price'])


chain = {}
store = pd.HDFStore(h5_name)
for i,row in keys_df.iterrows():
	link = {}
	link['raw_data'] = store[row['raw_data_key']]
	link['spot_price'] = float(store[row['spot_price']].iloc[0])
	chain[row['date']] = link
store.close()

matrix_chain = {}
dates = keys_df['date']
for i,date in enumerate(dates):
	raw_data = chain[date]['raw_data']
	spot = chain[date]['spot_price']
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
	T = vol_matrix.columns.tolist()
	K = vol_matrix.index.tolist()

	cols_to_map = [
	        'contractID', 'symbol', 'expiration', 'type', 'last', 'mark',
	        'bid', 'bid_size', 'ask', 'ask_size', 'volume', 'open_interest', 'date',
	        'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
	        'spot_price', 'moneyness'
	]
	for col in cols_to_map:
	    for i,row in hottest_contracts.iterrows():
	        hottest_contracts.at[i,col] = indexed.loc[(row['k'],row['t']),col]
	        
	hottest_contracts = hottest_contracts.rename(
	    columns={'t':'days_to_maturity','k':'strike_price'}).copy()
	matrix_chain[date] = vol_matrix

while True:
    try:
        with pd.HDFStore(h5_name) as store:
            for date, matrix in matrix_chain.items():
                h5_key = f"date_{date.replace('-','_')}/surface"
                store.put(h5_key, matrix, format='table', append=False)
                print(f"volatility surface stored for {date}")
        break
    except OSError as e:
        time.sleep(2)