import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt 
from model_settings import vanilla_pricer, asian_option_pricer, barrier_option_pricer, ms
from datetime import datetime
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





def generate_asian_options(s,r,g,fixing_frequencies,n_fixings,spread,calculation_datetime,kappa,theta,rho,eta,v0):

    K = np.unique(np.linspace(s*(1-spread),s*(1+spread),5).astype(int))

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



def row_generate_asian_options(row,fixing_frequencies,n_fixings,spread):
    return generate_asian_options(
        row['spot_price'],
        row['risk_free_rate'],
        row['dividend_rate'],
        fixing_frequencies,
        n_fixings,spread,
        row['calculation_date'],
        row['kappa'],
        row['theta'],
        row['rho'],
        row['eta'],
        row['v0']
    )



v0 = 0.005
kappa = 0.8
theta = 0.008
rho = 0.2
eta = 0.1

s = 100
k = 50
t = 360
w = 'put'
r = 0.04
g = 0.0
volatility = v0**0.5

calculation_datetime = datetime.today()

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


generate_asian_options(s,r,g,fixing_frequencies,n_fixings,spread,calculation_datetime,kappa,theta,rho,eta,v0)