from model_settings import ms
from pathlib import Path
import os
ms.find_root(Path())
ms.root = os.path.join(ms.root)
ms.collect_spx_calibrations()
params = ms.spx_calibrations
day = params.iloc[-3]


import numpy as np
import pandas as pd
from itertools import product

"""
barrier options
"""
# s, r, g = 5785, 0.040569, 0.01283
# kappa, theta, rho, eta, v0 = 3.676912, 0.255154, -1.0, 0.917628, 0.012088

# K = np.linspace(s*0.8,s*1.2,3,dtype=int)
# T = [30,60]
# B = np.linspace(s*0.5,s*1.2,4,dtype=int)
# R = [0]
# W = ['call','put']
# outin = ['Out','In']

# Xi = pd.DataFrame(
# 	product([s],[r],[g],[kappa],[theta],[rho],[eta],[v0],K,B,R,T,W,outin),
# 	columns=['spot_price', 'risk_free_rate', 'dividend_rate', 'kappa', 'theta', 'rho', 'eta', 'v0', 'strike_price', 'barrier', 'rebate', 'days_to_maturity', 'w','outin']
# )

# Xi['updown'] = Xi.apply(lambda row: 'Down' if row['barrier'] < row['spot_price'] else 'Up', axis=1)
# Xi['barrier_type_name'] = Xi['updown']+Xi['outin']
# Xi = Xi.drop(columns=['updown','outin'])


# from quantlib_pricers import barriers
# prices = pd.DataFrame(barriers.df_barrier_price(Xi))
# Xi[prices.columns] = prices

# k = 8
# dots = pd.DataFrame(np.tile(' . . . ',(1,len(Xi.columns))),columns=Xi.columns)
# table = pd.concat([Xi.head(k),dots,Xi.tail(k)])
# print(table)
# print()
# print(table.to_latex())


"""
asian options
"""


s, r, g = 5785, 0.040569, 0.01283
kappa, theta, rho, eta, v0 = 3.676912, 0.255154, -1.0, 0.917628, 0.012088
K = np.linspace(s*0.7,s*1.3,3,dtype=int)
past_fixings = [0]
fixing_frequencies = [7,28,84]
maturities = [7,28,84]
feature_list = []
for i,t in enumerate(maturities):
    for a in fixing_frequencies[:i+1]:
        df = pd.DataFrame(
            product(
            	[s],[r],[g],[kappa],[theta],[rho],[eta],[v0],
            	K,[t],[a],[0],
            	['geometric','arithmetic'],['call','put']
            ),
            columns = [
                'spot_price','risk_free_rate','dividend_rate',
                'kappa','theta','rho','eta','v0',
                'strike_price','days_to_maturity',
                'fixing_frequency','past_fixings',
                'averaging_type','w'
            ]
        )
        feature_list.append(df)
Xi = pd.concat(feature_list,ignore_index=True)

Xi['n_fixings'] = Xi['days_to_maturity']//Xi['fixing_frequency']
from quantlib_pricers import asians
prices = pd.DataFrame(asians.df_asian_option_price(Xi))
Xi[prices.columns] = prices
Xi = Xi.drop(columns='n_fixings')
k = 8
dots = pd.DataFrame(np.tile(' . . . ',(1,len(Xi.columns))),columns=Xi.columns)
table = pd.concat([Xi.head(k),dots,Xi.tail(k)])
print(table)
print()
print(table.to_latex())

