# -*- coding: utf-8 -*-
"""

Created on Fri Oct 11 19:42:05 2024

@author: boomelage


"""
import os
import sys
from pathlib import Path
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = str(Path().resolve().parent)
os.chdir(current_dir)
sys.path.append(parent_dir)

def parse_date(s):
    return s[s.find('_',0)+1:s.find('/',1)].replace('_','-')

symbol = 'SPY'
h5_name = f'alphaVantage {symbol}.h5'
while True:
    try:
        with pd.HDFStore(h5_name) as store:
            keys = store.keys()
            keys = pd.Series([key for key in keys if key.find('raw_data')!=-1])
        break
    except Exception as e:
        print(e)
        print('\nretrying in...')
        for i in range(2):
            print(2-i)
    finally:
        store.close()


dates = pd.Series([parse_date(key) for key in keys])

print(f"available dates:\n{dates}")
