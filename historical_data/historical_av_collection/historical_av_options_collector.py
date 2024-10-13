# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:02:17 2024

@author: boomelage
"""

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
import pandas as pd
from model_settings import ms
import requests
from datetime import datetime
import time
from historical_alphaVantage_collection import collect_av_link
from historical_av_underlying_fetcher import spots, symbol

print(spots)
for i in range(len(spots)):
    date = spots.iloc[i].name
    spot = spots.iloc[i]
    printdate = datetime.strptime(date, '%Y-%m-%d').strftime('%A, %Y-%m-%d')
    while True:
        try:
            raw_data = collect_av_link(date,spot,symbol)
            with pd.HDFStore(f'alphaVantage {symbol}.h5') as store:
                store.put(
                    f"date_{date.replace('-','_')}/raw_data",
                    raw_data,
                    format='table',
                    append=False
                )
            print(f'collecting: {printdate}')
            break
        except OSError:
            print(OSError)
            print('retrying in...')
            for i in range(2):
                print(2-i)
        except Exception as e:
            print(e)
            print(printdate)
            continue
        finally:
            store.close()
            print('collected')
