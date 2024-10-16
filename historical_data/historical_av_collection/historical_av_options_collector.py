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
import QuantLib as ql
import time
from historical_av_underlying_fetcher import spots, symbol


def collect_av_link(date,spot,symbol):
    options_url = str(
        "https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&"
        f"symbol={symbol}"
        f"&date={date}"
        f"&apikey={ms.av_key}"
              )
    r = requests.get(options_url)
    data = r.json()
    raw_data = pd.DataFrame(data['data'])
    return raw_data


for i in range(len(spots)):
    date = spots.iloc[i].name
    spot = spots.iloc[i]
    datetimedate = datetime.strptime(date, '%Y-%m-%d')
    ql_date = ql.Date(datetimedate.day,datetimedate.month,datetimedate.year)
    printdate = str(datetimedate.strftime('%A, ') + str(ql_date))
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
                store.put(
                    f"date_{date.replace('-','_')}/spot_price",
                    pd.Series(float(spot.iloc[0])),
                    format='fixed',
                    append=False
                )
            print(f'collecting: {date}')
            break
        except Exception as e:
            print(e)
            time.sleep(2)
        finally:
            store.close()
            print(f'collected {printdate}')
