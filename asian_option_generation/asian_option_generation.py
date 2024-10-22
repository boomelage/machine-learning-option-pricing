import os
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from datetime import datetime

class asian_option_feature_generation():
    def generate_asian_option_features(self,s,r,g,calculation_datetime,kappa,theta,rho,eta,v0):
        kupper = int(s*(1+0.5))
        klower = int(s*(1-0.5))
        K = np.arange(klower,kupper,5)

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
            1,7,28,84,168,336,672
        ]

        maturities = [
            7,28,84,168,336,672
        ]
        features = []
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
                features.append(df)

        for f in fixing_frequencies:
            features.append(
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
        return pd.concat(features,ignore_index=True)

    def row_generate_asian_option_features(self,row):
        return self.generate_asian_option_features(
            row['spot_price'],
            row['risk_free_rate'],
            row['dividend_rate'],
            row['date'],
            row['kappa'],
            row['theta'],
            row['rho'],
            row['eta'],
            row['v0']
        )


    def df_generate_asian_option_features(self,df):
        return Parallel()(delayed(self.row_generate_asian_option_features)(row) for _, row in df.iterrows())


from pathlib import Path
calibrations = pd.read_csv([file for file in os.listdir(str(Path().resolve())) if file.find('av_recalibrated')!=-1][0]).iloc[:,1:]
calibrations = calibrations.rename(columns={'calculation_date':'date'})
calibrations['date'] = pd.to_datetime(calibrations['date'],format='%Y-%m-%d')
calibrations['risk_free_rate'] = 0.04

gen = asian_option_feature_generation()

features = gen.df_generate_asian_option_features(calibrations)
features = pd.concat(features,ignore_index=True)
print(features)