# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:02:17 2024

@author: boomelage
"""
import pandas as pd
import requests
from model_settings import ms

def collect_av_link(date,symbol):
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