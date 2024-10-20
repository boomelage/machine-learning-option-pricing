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
        K = np.arange(klower,kupper,7)

        W = [
            'call',
            'put'
        ]

        types = [
            'arithmetic',
            'geometric'
        ]


        past_fixings = [0]

        fixing_frequencies = [30,60,90,180,360]
        max_maturity = fixing_frequencies[-1]
        option_features = []
        for f in fixing_frequencies:
            max_periods = np.arange(f,max_maturity+1,f)
            n_fixings = len(max_periods)
            for i, tenor in enumerate(max_periods):
                n_fixings = i+1
                periods = max_periods[:i+1]

                features = pd.DataFrame(
                    product(
                        [s],
                        K,
                        [r],
                        [g],
                        W,
                        types,
                        [f],
                        [n_fixings],
                        past_fixings,
                        [kappa],
                        [theta],
                        [rho],
                        [eta],
                        [v0],
                        [calculation_datetime],
                        [tenor]
                    ),
                    columns = [
                        'spot_price','strike_price','risk_free_rate','dividend_rate','w',
                        'averaging_type','fixing_frequency','n_fixings','past_fixings',
                        'kappa','theta','rho','eta','v0','calculation_date','days_to_maturity'
                    ]
                )
                option_features.append(features)
        return pd.concat(option_features,ignore_index=True)

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
calibrations = pd.read_csv([file for file in os.listdir(str(Path().resolve())) if file.find('SPY calibrated')!=-1][0]).iloc[:,1:]
calibrations['date'] = pd.to_datetime(calibrations['date'],format='%Y-%m-%d')

gen = asian_option_feature_generation()
option_features = gen.df_generate_asian_option_features(calibrations)
option_features = pd.concat(option_features,ignore_index=True)