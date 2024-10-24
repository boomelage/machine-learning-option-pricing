import os
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from model_settings import ms, asian_option_pricer
aop = asian_option_pricer()
root = Path(__file__).resolve().parent.parent.parent.parent.parent


underlying_product = ms.cboe_spx_asians


tag = underlying_product['calibrations_filetag']
calibrations_dir = underlying_product['calibrations_dir']
output_dump = underlying_product['dump']

calibrations_dir = os.path.join(root,calibrations_dir)
file = [f for f in os.listdir(calibrations_dir) if f.find(tag)!=-1][0]
filepath = os.path.join(calibrations_dir,file)

output_dir = os.path.join(root,output_dump)
computed_outputs = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])

calibrations = pd.read_csv(filepath).iloc[computed_outputs:,1:]
calibrations['calculation_date'] = pd.to_datetime(calibrations['calculation_date'],format='%Y-%m-%d')
calibrations['risk_free_rate'] = 0.04

print(f"\n{calibrations}")

bar = tqdm(total = calibrations.shape[0])
def generate_asian_option_features(s,r,g,calculation_datetime,kappa,theta,rho,eta,v0):
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
        1,7,28,84,168,336,#672
    ]

    maturities = [
        7,28,84,168,336,#672
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
                    [calculation_datetime],
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

    for f in fixing_frequencies:
        feature_list.append(
            pd.DataFrame(
                product(
                    [s],
                    K,
                    [f],
                    [1],
                    [f],
                    [0],
                    ['geometric','arithmetic'],
                    ['call','put'],
                    [r],
                    [g],
                    [calculation_datetime],
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
        )
    features = pd.concat(feature_list,ignore_index=True)
    features['asian'] = aop.df_asian_option_price(features)
    features.to_csv(os.path.join(output_dir,f"{calculation_datetime.strftime('%Y-%m-%d')} {tag} asian options.csv"))
    bar.update(1)

def row_generate_asian_option_features(row):
    generate_asian_option_features(
        row['spot_price'],
        row['risk_free_rate'],
        row['dividend_rate'],
        row['calculation_date'],
        row['kappa'],
        row['theta'],
        row['rho'],
        row['eta'],
        row['v0']
    )



import time
start = time.time()
calibrations.apply(row_generate_asian_option_features,axis=1)
bar.close()
end = time.time()
runtime = start-end