# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from datetime import datetime
from model_settings import barrier_option_pricer
barp = barrier_option_pricer()

script_dir = Path(__file__).resolve().parent.absolute()
datadir =  os.path.join(script_dir.parent.parent.parent.parent,'OneDrive - rsbrc','DATA','calibrated','bloomberg','SPX')
file = [f for f in os.listdir(datadir) if f.endswith('.csv')][0]
filepath = os.path.join(datadir,file)

output_dir = os.path.join(Path(datadir).parent.parent.parent,'generated','bloomberg','barrier_options')
if not os.path.exists(output_dir):
    os.mkdir(os.path.join(output_dir))


def generate_barrier_features(s, K, T, barriers, updown, OUTIN, W):
    barrier_features = pd.DataFrame(
        product([s], K, barriers, T, [updown], OUTIN, W),
        columns=[
            'spot_price', 'strike_price', 'barrier', 'days_to_maturity',
            'updown', 'outin', 'w'
        ]
    )
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    return barrier_features

df = pd.read_csv(filepath).iloc[:,1:]
bar = tqdm(total=df.shape[0])

def row_generate_barrier_features(row):
    s = row['spot_price']
    
    date = row['date']
    calculation_datetime = datetime.strptime(date,'%Y-%m-%d')
    date_print = datetime(
        calculation_datetime.year,
        calculation_datetime.month,
        calculation_datetime.day
        ).strftime('%A, %Y-%m-%d')

    r = 0.04 
    rebate = 0.
    step = 1
    atm_spread = 1
    r = 0.04
    K = np.linspace(
        s*0.9,
        s*1.1,
        9
        )
    T = [
        60,
        90,
        180,
        360,
        540,
        720
        ]
    OUTIN = ['Out','In']
    W = ['call','put']
        
    
    barriers = np.linspace(
        s*0.5,s*0.99,
        5
        ).astype(float).tolist()
    down_features = generate_barrier_features(
        s, K, T, barriers, 'Down', OUTIN, W)
    
    
    barriers = np.linspace(
        s*1.01,s*1.5,
        5
        ).astype(float).tolist()
    up_features = generate_barrier_features(
        s, K, T, barriers, 'Up', OUTIN, W)

    
    features = pd.concat(
        [down_features,up_features],
        ignore_index = True
        )
    features['rebate'] = rebate
    features['dividend_rate'] = row['dividend_rate']
    features['risk_free_rate'] = r
    features.loc[:,'theta'] = np.tile(row['theta'],features.shape[0])
    features.loc[:,'kappa'] = np.tile(row['kappa'],features.shape[0])
    features.loc[:,'rho'] = np.tile(row['rho'],features.shape[0])
    features.loc[:,'eta'] = np.tile(row['eta'],features.shape[0])
    features.loc[:,'v0'] = np.tile(row['v0'],features.shape[0])
    features['calculation_date'] = calculation_datetime
    features['barrier_price'] = barp.df_barrier_price(features)
    features['calculation_date'] = date

    features.to_csv(os.path.join(output_dir,f'{date} bloomberg SPX barrier options.csv'))
    bar.update(1)


df.apply(row_generate_barrier_features,axis=1)

bar.close()