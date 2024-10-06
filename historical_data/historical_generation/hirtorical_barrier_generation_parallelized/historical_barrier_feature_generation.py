# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import QuantLib as ql
from tqdm import tqdm
from itertools import product
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(grandgrandparent_dir)
from settings import model_settings
from data_query import dirdatacsv
ms = model_settings()
os.chdir(current_dir)



    

def generate_barrier_type(
        s,T,updown,r,g,rebate,
        theta,kappa,rho,eta,v0,
        calculation_date):
    
    if updown == 'Up':
        udw = 1
    elif updown == 'Down':
        udw = -1
    else:
        raise ValueError('invalid Up/Down flag')
    
    K = np.linspace(
        s*0.9,
        s*1.1,
        50
        )
    
    barriers = np.linspace(
        s*(1+udw*0.01),
        s*(1+udw*0.5),
        5
        ).astype(float).tolist()
    
    barrier_features =  pd.DataFrame(
        product(
            [s],
            K,
            barriers,
            T,
            [updown],
            ['Out','In'],
            ['call','put'],
            [rebate],
            [calculation_date]
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'barrier',
            'days_to_maturity',
            'updown',
            'outin',
            'w',
            'rebate',
            'calculation_date'
                  ]
        )
    
    barrier_features['barrier_type_name'] = \
        barrier_features['updown'] + barrier_features['outin']
    
    barrier_features['dividend_rate'] = g
    barrier_features['risk_free_rate'] = r
    barrier_features['theta'] = theta
    barrier_features['kappa'] = kappa
    barrier_features['rho'] = rho
    barrier_features['eta'] = eta
    barrier_features['v0'] = v0
    return barrier_features



def process_row(row,T):
    s = row['spot_price']
    r = row['risk_free_rate']
    g = row['dividend_rate']
    rebate = row['rebate']
    theta = row['theta']
    kappa = row['kappa']
    rho = row['rho']
    eta = row['eta']
    v0 = row['v0']
    calculation_datetime = row['date']
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year
        )
    up_features = generate_barrier_type(
        s, T, 'Up', r, g, rebate, 
        theta, kappa, rho, eta, v0,
        calculation_date
        )
    down_features = generate_barrier_type(
        s, T, 'Down', r, g, rebate, 
        theta, kappa, rho, eta, v0,
        calculation_date
        )
    barrier_features = pd.concat(
        [up_features, down_features], ignore_index=True)
    
    return barrier_features






"""
###########
# routine #
###########
"""

historical_calibrated = pd.read_csv(dirdatacsv()[0])
historical_calibrated = historical_calibrated.iloc[:1,1:].copy(
    ).reset_index(drop=True)
historical_calibrated['date'] = pd.to_datetime(historical_calibrated['date'])
historical_calibrated['risk_free_rate'] = 0.04
historical_calibrated['rebate'] = 0.0

T = [
    60,
    # 90,
    # 180,
    # 360,
    # 540,
    # 720
    ]


barrier_feature_list = historical_calibrated.apply(
    process_row, axis=1).tolist()

barrier_features = pd.concat(barrier_feature_list,ignore_index=True)

barrier_features['barrier_price'] = ms.vector_barrier_price(barrier_features)

barrier_features.groupby('calculation_date')