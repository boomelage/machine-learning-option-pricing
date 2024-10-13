import requests
import pandas as pd
from datetime import datetime
from model_settings import ms
from historical_av_key_collector import keys_df, symbol, available_dates
"""
https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo
"""
symbol='SPY'
url = str(
	'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+
	symbol+'&outputsize=full&apikey='+
	ms.av_key)
r = requests.get(url)
spots = pd.Series(pd.DataFrame(r.json()['Time Series (Daily)']).transpose()['4. close'].squeeze())
spots = pd.to_numeric(spots,errors='coerce').reset_index().rename(columns={'index':'date','4. close':'spot_price'})
spots = spots.set_index('date')
historical_spots = spots.copy()
try:
	spots = spots[~spots.index.isin(available_dates)]
	spots = spots[spots.index != '2020-06-23']
except Exception as e:
	pass
print(f"\ndata to collect for {symbol}:\n{spots}")
