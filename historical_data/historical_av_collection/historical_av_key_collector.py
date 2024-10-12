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

while True:
    try:
        store = pd.HDFStore(os.path.join(current_dir,'alphaVantage vanillas.h5'))
        keys = store.keys()
        contract_keys = pd.Series([key for key in keys if key.find('hottest_contracts')!=-1])
        raw_data_keys = pd.Series([key for key in keys if key.find('raw_data')!=-1])
        surface_keys = pd.Series([key for key in keys if key.find('surface')!=-1])
        calibration_keys = pd.Series([key for key in keys if key.find('calibration')!=-1])
        prices_securities_keys = pd.Series([key for key in keys if key.find('priced_securities')!=-1])
        break
    except Exception as e:
        print(e)
        print('\nretrying in...')
        for i in range(2):
            print(2-i)
    finally:
        store.close()

keys_df = pd.DataFrame(
    {
     'contract_key':contract_keys,
     'raw_data_key':raw_data_keys,
     'surface_key':surface_keys,
     'calibration_key':calibration_keys,
     'priced_securities_key':prices_securities_keys
     }    
)

keys_df['date'] = keys_df['surface_key'].str.extract(r'date_(\d{4}_\d{2}_\d{2})')[0]

keys_df['date'] = pd.to_datetime(keys_df['date'], format='%Y_%m_%d')

print(keys_df)