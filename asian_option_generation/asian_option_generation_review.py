import os
import time
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
from datetime import datetime
from pathlib import Path
import QuantLib as ql

class asian_option_pricer():
    def __init__(self,seed=123):
        if seed == None:
            self.seed = 0
        else:
            self.seed = seed
        self.rng = "pseudorandom" # could use "lowdiscrepancy"
        self.numPaths = 100000
        print('Asian option pricer initialized')
        print('day counter is QuantLib 30/360 USA\n')

    def asian_option_price(self,s,k,r,g,w,averaging_type,n_fixings,fixing_frequency,past_fixings,kappa,theta,rho,eta,v0,calculation_datetime):
        s = float(s)
        k = float(k)
        r = float(r)
        g = float(g)

        
        if w == 'call':
            option_type = ql.Option.Call 
        elif w == 'put':
            option_type = ql.Option.Put
        t = n_fixings*fixing_frequency
        

        calculation_date = ql.Date(calculation_datetime.day,calculation_datetime.month,calculation_datetime.year)
        
        periods = np.arange(fixing_frequency,t+1,fixing_frequency).astype(int)
        fixing_periods = [ql.Period(int(p),ql.Days) for p in periods]
        fixing_dates = [calculation_date + p for p in fixing_periods]

        expiration_date = calculation_date + fixing_periods[-1]


        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),ql.Thirty360(ql.Thirty360.USA)))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),ql.Thirty360(ql.Thirty360.USA)))

        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(s)), v0, kappa, theta, eta, rho)
        hestonModel = ql.HestonModel(hestonProcess)
        vanillaPayoff = ql.PlainVanillaPayoff(option_type, float(k))
        europeanExercise = ql.EuropeanExercise(expiration_date)
        
        if averaging_type == 'geometric':
            geometric_engine = ql.MCDiscreteGeometricAPHestonEngine(hestonProcess, self.rng, requiredSamples=self.numPaths,seed=self.seed)
            geometricAverage = ql.Average().Geometric
            geometricRunningAccumulator = 1.0
            discreteGeometricAsianOption = ql.DiscreteAveragingAsianOption(
                geometricAverage, geometricRunningAccumulator, past_fixings,
                fixing_dates, vanillaPayoff, europeanExercise)
            discreteGeometricAsianOption.setPricingEngine(geometric_engine)
            geometric_price = float(discreteGeometricAsianOption.NPV())
            return geometric_price
            
        else:
            arithmetic_engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, self.rng, requiredSamples=self.numPaths)
            arithmeticAverage = ql.Average().Arithmetic
            arithmeticRunningAccumulator = 0.0
            discreteArithmeticAsianOption = ql.DiscreteAveragingAsianOption(
                arithmeticAverage, arithmeticRunningAccumulator, past_fixings, 
                fixing_dates, vanillaPayoff, europeanExercise)
            discreteArithmeticAsianOption.setPricingEngine(arithmetic_engine)
            arithmetic_price = float(discreteArithmeticAsianOption.NPV())
            return arithmetic_price

    def row_asian_option_price(self,row):
        return  self.asian_option_price(
            row['spot_price'],
            row['strike_price'],
            row['risk_free_rate'],
            row['dividend_rate'],
            row['w'],
            row['averaging_type'],
            row['fixing_frequency'],
            row['n_fixings'],
            row['past_fixings'],
            row['kappa'],
            row['theta'],
            row['rho'],
            row['eta'],
            row['v0'],
            row['calculation_date']
        )

    def df_asian_option_price(self, df):
        max_jobs = os.cpu_count() // 4

        max_jobs = max(1, max_jobs)

        return Parallel(n_jobs=max_jobs)(delayed(self.row_asian_option_price)(row) for _, row in df.iterrows())


def generate_asian_option_features(s,r,g,calculation_datetime,kappa,theta,rho,eta,v0):
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

def row_generate_asian_option_features(row):
    return generate_asian_option_features(
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


def df_generate_asian_option_features(df):
    return Parallel()(delayed(row_generate_asian_option_features)(row) for _, row in df.iterrows())




"""

usage


"""

aop = asian_option_pricer()




calibrations = pd.read_csv([file for file in os.listdir(str(Path().resolve())) if file.find('SPY calibrated')!=-1][0]).iloc[:,1:]
calibrations['date'] = pd.to_datetime(calibrations['date'],format='%Y-%m-%d')

option_features = df_generate_asian_option_features(calibrations)





# print(calibrations.dtypes)
# calibrations['risk_free_rate'] = 0.04

# day = {}
# for i in calibrations.columns:
#     day[i] = calibrations[i].iloc[0]
#     print(f"{i} {calibrations[i].iloc[0]}")
# s = day['spot_price']
# g = day['dividend_rate']
# calculation_datetime = day['date']
# theta = day['theta']
# kappa = day['kappa']
# rho = day['rho']
# eta = day['eta']
# v0 = day['v0']
# r = day['risk_free_rate']


# kupper = int(s*(1+0.5))
# klower = int(s*(1-0.5))
# K = np.arange(klower,kupper,7)

# W = [
#     'call',
#     'put'
# ]

# types = [
#     # 'arithmetic',
#     'geometric'
# ]

# fixing_frequencies = [
#     # 1,
#     # 7,
#     30,
#     # 90,
# ]

# n_fixings = [
#     1,
#     # 3,
#     # 6,12
# ]

# past_fixings = [0]


# features = pd.DataFrame(
#     product(
#         [s],
#         K,
#         [r],
#         [g],
#         W,
#         types,
#         fixing_frequencies,
#         n_fixings,
#         past_fixings,
#         [kappa],
#         [theta],
#         [rho],
#         [eta],
#         [v0],
#         [calculation_datetime]
#     ),
#     columns = [
#         'spot_price','strike_price','risk_free_rate','dividend_rate','w',
#         'averaging_type','fixing_frequency','n_fixings','past_fixings',
#         'kappa','theta','rho','eta','v0','calculation_date'
#     ]
# )

# # features['asian_price'] = aop.df_asian_option_price(features)

# print(features)

