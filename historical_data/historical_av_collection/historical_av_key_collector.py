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

symbol = 'SPY'
h5_name = f'alphaVantage {symbol}.h5'
while True:
    try:
        with pd.HDFStore(os.path.join(current_dir,h5_name)) as store:
            keys = store.keys()
            raw_data_keys = pd.Series([key for key in keys if key.find('raw_data')!=-1])
            surface_keys = pd.Series([key for key in keys if key.find('surface')!=-1])
            spot_price_keys = pd.Series([key for key in keys if key.find('spot_price')!=-1])
            calibration_keys = pd.Series([key for key in keys if key.find('heston_calibration/calibration_results')!=-1])
            parameter_keys = pd.Series([key for key in keys if key.find('heston_calibration/heston_parameters')!=-1])
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
     'raw_data_key':raw_data_keys,
     'surface_key':surface_keys,
     'spot_price':spot_price_keys,
     'calibration_key':calibration_keys,
     'parameter_key':parameter_keys
     }    
)


try:
    available_dates = keys_df['raw_data_key'].copy().str.extract(r'date_(\d{4}_\d{2}_\d{2})')[0].squeeze()
    available_dates = available_dates.str.replace('_','-')
    keys_df['date'] = available_dates
except Exception as e:
    pass

keys_df = keys_df.dropna(subset='raw_data_key')
print(f"\ncontracts data available for {symbol}:\n{available_dates}\n")
pd.set_option("display.max_columns",None)
print(keys_df)
print(keys_df.dtypes)
pd.reset_option("display.max_columns")



# try:
#     with pd.HDFStore(h5_name) as store:
#         for i,row in keys_df.iterrows():
#             del store[row['calibration_key']]
#             del store[row['parameter_key']]
#             print(f"deleted {row['calibration_key']}")
#             print(f"deleted {row['parameter_key']}")

#     store.close()

# except Exception as e:
#     raise(e)
#     pass
