# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:41:51 2024

@author: boomelage
"""

from model_settings import ms
import requests
import pandas as pd

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