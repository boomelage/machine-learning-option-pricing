import pandas as pd
import numpy as np
from historical_av_key_collector import h5_name,dates,symbol
from historical_av_underlying_fetcher import spots

with pd.HDFStore(h5_name) as store:
	keys = store.keys()
store.close()

categories = pd.Series([key[key.rfind('/')+1:] for key in keys]).drop_duplicates()
raw_data_keys = pd.Series([key for key in keys if key.find('raw_data')!=-1])

keys_df = pd.DataFrame(
	np.empty((len(raw_data_keys),len(categories)),dtype=object),
	columns = categories
)
keys_df['raw_data'] = raw_data_keys
keys_df['date'] = dates

for i,row in keys_df.iterrows():
	key = row['raw_data']
	locator = key[:key.find('/',1)]
	row_keys = [key for key in keys if key.find(locator)!=-1]
	for cat in categories:
		try:
			keys_df.at[i,cat] = [key for key in row_keys if key.find(cat)!=-1][0]
		except Exception:
			keys_df.at[i,cat] = np.nan

pd.set_option("display.max_columns",None)
keys_df = keys_df[['date','raw_data', 'spot_price', 'surface', 'calibration_results',
       'heston_parameters']]
print(f"{keys_df[['date','raw_data']]}\n")

# deletion_keys = keys_df[
# 	(
# 		(keys_df['date'].str.contains('2008'))|
# 		(keys_df['date'].str.contains('2009'))|
# 		(keys_df['date'].str.contains('2010'))|
# 		(keys_df['date'].str.contains('2011'))|
# 		(keys_df['date'].str.contains('2012'))
# 	)
# ]

with pd.HDFStore(h5_name) as store:
	for i,row in keys_df.iterrows():
		for col in keys_df.columns[1:]:
			try:
				calibration = store[row['calibration_results']]
				if np.mean(np.abs(calibration['error']))<0.5:
					del store[row[col]]
					print(f"deleted {row[col]}")
				else:
					pass
			except Exception as e:
				print(e)
				pass
store.close()

