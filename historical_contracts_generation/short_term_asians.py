import os
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from model_settings import ms
from quantlib_pricers import asian_option_pricer
pd.set_option('display.max_columns',None)


aop = asian_option_pricer()
root = Path(__file__).resolve().parent.parent.parent.parent.parent
underlying_product = ms.cboe_spx_short_term_asians
ms.find_root(Path(__file__).resolve())
ms.collect_spx_calibrations()
df = ms.spx_calibrations
df['calculation_date'] = pd.to_datetime(df['calculation_date'],format='mixed')
df = df.sort_values(by='calculation_date',ascending=False).reset_index(drop=True)

tag = underlying_product['calibrations_filetag']
calibrations_dir = underlying_product['calibrations_dir']
output_dump = underlying_product['dump']

output_dir = os.path.join(root,output_dump)
computed_outputs = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])

df = df.iloc[computed_outputs:].copy()

print(f"\n{df}")


bar = tqdm(total = df.shape[0])
def row_generate_asian_option_features(row):
    s = row['spot_price']
    r = row['risk_free_rate']
    g = row['dividend_rate']
    calculation_date = row['calculation_date']
    kappa = row['kappa']
    theta = row['theta']
    rho = row['rho']
    eta = row['eta']
    v0  = row['v0']
    kupper = int(s*(1+0.5))
    klower = int(s*(1-0.5))
    K = np.linspace(klower,kupper,5)

    W = [
        'call',
        'put'
    ]

    types = [
        'arithmetic',
        'geometric'
    ]


    past_fixings = [0]

    fixing_frequencies = [
        7,28,84
    ]

    maturities = [
        7,28,84
    ]


    feature_list = []
    for i,t in enumerate(maturities):
        for f in fixing_frequencies[:i+1]:
            n_fixings = [t/f]
            df = pd.DataFrame(
                product(
                    [s],
                    K,
                    [t],
                    n_fixings,
                    [f],
                    [0],
                    ['geometric','arithmetic'],
                    ['call','put'],
                    [r],
                    [g],
                    [calculation_date],
                    [kappa],
                    [theta],
                    [rho],
                    [eta],
                    [v0]
                ),
                columns = [
                    'spot_price','strike_price','days_to_maturity',
                    'n_fixings','fixing_frequency','past_fixings','averaging_type','w',
                    'risk_free_rate','dividend_rate','calculation_date',
                    'kappa','theta','rho','eta','v0'
                ]
            )
            feature_list.append(df)
    features = pd.concat(feature_list,ignore_index=True)
    features['date'] = calculation_date.floor('D')
    features['asian_price'] = aop.df_asian_option_price(features)
    features.to_csv(os.path.join(output_dir,f"{calculation_date.strftime('%Y-%m-%d_%H%M%S%f')}_{(str(int(s*100))).replace('_','')}  {tag} short-term asian options.csv"))
    bar.update(1)


import time
start = time.time()
df.apply(row_generate_asian_option_features,axis=1)
bar.close()
end = time.time()
runtime = end-start
print(f"\ncpu: {runtime}\n")