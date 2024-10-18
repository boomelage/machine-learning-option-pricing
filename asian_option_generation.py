import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt 
from model_settings import vanilla_pricer, asian_option_pricer, barrier_option_pricer, ms
from datetime import datetime
from pathlib import Path
vp = vanilla_pricer.vanilla_pricer()
bop = barrier_option_pricer.barrier_option_pricer()
aop = asian_option_pricer.asian_option_pricer()



# help(bop.barrier_price)
# help(aop.asian_option_price)

# black_scholes = vp.numpy_black_scholes(s,k,t,r,volatility,w)
# heston = vp.heston_price(s,k,t,r,g,w,kappa,theta,rho,eta,v0,calculation_datetime)
# barrier = bop.barrier_price(s, k, t, r, g, calculation_datetime, w, 'DownIn', s, 0.0, kappa, theta, rho, eta, v0)
# asian = aop.asian_option_price(s,k,r,g,w,'geometric',1,t,0,kappa,theta,rho,eta,v0,calculation_datetime)
# print(f"\nblack scholes: {black_scholes}\nheston: {heston}\nbarrier: {barrier}\nasian: {asian}")





def generate_asian_options(s,r,g,fixing_frequencies,n_fixings,n_strikes,spread,calculation_datetime,kappa,theta,rho,eta,v0):

    K = np.unique(np.linspace(s*(1-spread),s*(1+spread),n_strikes).astype(int))

    W = [
        'call',
        'put'
    ]

    types = [
        'arithmetic',
        'geometric'
    ]

    past_fixings = [0]
    features = pd.DataFrame(
        product(
            [s],
            K,
            [r],
            [g],
            W,
            types,
            fixing_frequencies,
            n_fixings,
            past_fixings,
            [kappa],
            [theta],
            [rho],
            [eta],
            [v0],
            [calculation_datetime]
        ),
        columns = [
            'spot_price','strike_price','risk_free_rate','dividend_rate','w',
            'averaging_type','fixing_frequency','n_fixings','past_fixings',
            'kappa','theta','rho','eta','v0','calculation_date'
        ]
    )
    features['days_to_maturity'] = features['n_fixings']*features['fixing_frequency']
    features['vanilla'] = vp.df_heston_price(features)
    features['asian_price'] = aop.df_asian_option_price(features)

    # features['difference'] = features['vanilla']-features['asian_price']

    features = features[
        [
            # 'difference',
            'vanilla', 'asian_price','spot_price', 'strike_price', 'risk_free_rate', 'dividend_rate', 'w',
            'averaging_type', 'fixing_frequency', 'n_fixings', 'past_fixings',
            'kappa', 'theta', 'rho', 'eta', 'v0', 'calculation_date',
            'days_to_maturity'
        ]
    ]

    # features['moneyness'] = ms.vmoneyness(features['spot_price'],features['strike_price'],features['w'])
    # features = features.sort_values(by='moneyness',ascending=True).reset_index(drop=True)
    key = calculation_datetime.strftime('date_%Y_%m_%d/')
    with pd.HDFStore(r'asians.h5') as store:
        store.put(key,features,format='table',append=False)
    store.close()




fixing_frequencies = [
    1,
    7,
    30,
    180,
    360
]

n_fixings = [
    1,
    5,
    10
]
spread = 0.5


calibrations = pd.read_csv([file for file in os.listdir(str(Path().resolve())) if file.find('calibrated')!=-1][0]).iloc[:,1:]

calibrations['date'] = pd.to_datetime(calibrations['date'],format='%Y-%m-%d')
calibrations['risk_free_rate'] = 0.04
n_strikes = 5
from tqdm import tqdm
bar = tqdm(total=calibrations.shape[0])
for i,row in calibrations.iterrows():
    generate_asian_options(
        row['spot_price'],
        row['risk_free_rate'],
        row['dividend_rate'],
        fixing_frequencies,n_fixings,
        n_strikes,spread,
        row['date'],
        row['kappa'],
        row['theta'],
        row['rho'],
        row['eta'],
        row['v0']
    )
    bar.update(1)
bar.close()
