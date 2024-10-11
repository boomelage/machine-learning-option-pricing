# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:42:05 2024

@author: boomelage
"""
from pathlib import Path
import os
import sys
from tqdm import tqdm
current_dir = str(Path().resolve())
parent_dir = str(Path().resolve().parent)
os.chdir(current_dir)
sys.path.append(parent_dir)

import pandas as pd

while True:
	try:
		store = pd.HDFStore('alphaVantage vanillas.h5')
		keys = store.keys()
		contract_keys = [key for key in keys if key.find('hottest_contracts')!=-1]
		spots = pd.Series()
		bar = tqdm(total = len(contract_keys))
		for key in contract_keys:
			s = store[key]['spot_price'].unique()[0]
			date = store[key]['date'].unique()[0]
			spots[date] = float(s)
			bar.update(1)
		break
	except OSError:
		print(OSError)
		print('retrying in...')
		for i in range(5):
			print(5-i)
	finally:
		bar.close()
		store.close()


import matplotlib.pyplot as plt
plt.figure()
plt.plot(spots, color='purple')
plt.xticks(rotation=45)
plt.ylabel('spot price')
plt.title('SPY')
plt.show()
plt.clf()

earliest_date = spots.reset_index().describe().iloc[2,0].strftime('%Y-%m-%d')

from model_settings import ms
import requests

key = ms.av_key
symbol = 'SPY'
date = earliest_date


print(date)
underlying_url = str(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
    f"&symbol={symbol}&date={date}&outputsize=full&apikey={key}"
    )

spotr = requests.get(underlying_url)
spots = pd.DataFrame(spotr.json()['Time Series (Daily)']).T
spots = spots.astype(float)['4. close']

dates = spots.index.tolist()
pddates = pd.Series(dates)
collection_end_index = (pddates[pddates == date].index + 1).values[0]
dates = dates[collection_end_index:]

