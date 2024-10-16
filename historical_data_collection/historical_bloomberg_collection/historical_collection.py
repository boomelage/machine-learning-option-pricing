# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import sys
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

os.chdir(current_dir)
csvs = [file for file in os.listdir() if file.endswith('.csv')]

print(csvs)
"""
data collection and cleaning
"""

def collect_historical_data():
    historical_data = pd.DataFrame()

    price_counter = 0
    for file in csvs:
        raw = pd.read_csv(file)
        raw.columns = raw.loc[2]
        
        df = raw.loc[4:].reset_index(drop=True)
        
        df.columns
        
        df.rename(columns={
            'SPX Index' : 'date', 
            '30 Day IVOL at 100.0% Moneyness LIVE':'30D',
            '60 Day IVOL at 100.0% Moneyness LIVE':"60D",
            '3 Month 100% Moneyness Implied Volatility':"3M",
            '6 Month IVOL at 100.0% Moneyness LIVE':"6M",
            '1 Year 100% Moneyness Implied Volatility':"12M",
            '18 Month IVOL at 100.0% Moneyness LIVE':"18M",
            '24 Month IVOL at 100.0% Moneyness LIVE':"24M", 
            'BEst Div Yld':"dividend_rate",
            'End of Day Risk Free Rate Mid':"risk_free_rate", 
            'Mid Price':"spot_price",
            }, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y',errors='coerce')
        df = df.infer_objects(copy=False) 
        df[df.columns[1:]] = df[df.columns[1:]].astype(float)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        
        historical_data = pd.concat([historical_data,df],ignore_index=True)
        historical_data = historical_data.dropna(how='all',axis=0)
        historical_data = historical_data.dropna(how='all',axis=1)
        historical_data = historical_data.dropna(how='any',axis=0)
        price_counter += historical_data.shape[0]
        
    return historical_data

historical_data = collect_historical_data()

historical_data['date'] = pd.to_datetime(historical_data['date'])

os.chdir(current_dir)

pd.set_option("display.max_columns",None)

historical_data[
    ['30D', '60D', '3M', '6M', '12M', '18M', '24M', 'dividend_rate']
    ] = historical_data[
        ['30D', '60D', '3M', '6M', '12M', '18M', '24M', 'dividend_rate']
        ]/100

print(f"\nhistorical data:\n{historical_data.head(10)}")
