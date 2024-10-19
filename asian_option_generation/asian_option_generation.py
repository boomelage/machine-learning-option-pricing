import os
import time
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
from datetime import datetime
from pathlib import Path
from model_settings import vanilla_pricer, asian_option_pricer


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

        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),ql.Actual365Fixed()))

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
            
        elif averaging_type == 'arithmetic':
            arithmetic_engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, self.rng, requiredSamples=self.numPaths)
            arithmeticAverage = ql.Average().Arithmetic
            arithmeticRunningAccumulator = 0.0
            discreteArithmeticAsianOption = ql.DiscreteAveragingAsianOption(
                arithmeticAverage, arithmeticRunningAccumulator, past_fixings, 
                fixing_dates, vanillaPayoff, europeanExercise)
            discreteArithmeticAsianOption.setPricingEngine(arithmetic_engine)
            arithmetic_price = float(discreteArithmeticAsianOption.NPV())
            return arithmetic_price
        else:
            print("invalid Asian option averaging type out of 'arithmetic' and geometric'")
            pass

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
        max_jobs = os.cpu_count() // 3

        max_jobs = max(1, max_jobs)

        return Parallel(n_jobs=max_jobs)(delayed(self.row_asian_option_price)(row) for _, row in df.iterrows())



aop = asian_option_pricer()

def generate_asian_options(s,r,g,n_strikes,spread,calculation_datetime,kappa,theta,rho,eta,v0):

    K = np.unique(np.linspace(s*(1-spread),s*(1+spread),n_strikes).astype(int))

    W = [
        'call',
        'put'
    ]

    types = [
        'arithmetic',
        'geometric'
    ]

    fixing_frequencies = [
        # 1,
        # 7,
        30,
        # 90,
        # 180
    ]

    n_fixings = [
        1,
        5,
        10
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
    features['asian_price'] = aop.df_asian_option_price(features)

    datetag = calculation_datetime.strftime('%Y_%m_%d')
    filetag = datetime.today().strftime('%Y-%m-%d %H%M%S')
    filename = f"{datetag} asian options {filetag}.csv"
    filepath = os.path.join(str(Path().resolve()),'historical_asian_options',filename)
    features.to_csv(filepath)

def row_generate_asian_options(row):
    generate_asian_options(
        row['spot_price'],
        row['risk_free_rate'],
        row['dividend_rate'],
        row['n_strikes'],
        row['spread'],
        row['date'],
        row['kappa'],
        row['theta'],
        row['rho'],
        row['eta'],
        row['v0']
    )


def df_generate_asian_options(df):
    max_jobs = os.cpu_count() // 4

    max_jobs = max(1, max_jobs)

    Parallel(n_jobs=max_jobs)(delayed(row_generate_asian_options)(row) for _, row in df.iterrows())

spread = 0.5
n_strikes = 5

calibrations = pd.read_csv([file for file in os.listdir(str(Path().resolve())) if file.find('SPY calibrated')!=-1][0]).iloc[:,1:]
calibrations = calibrations.rename(columns = {'calculation_date':'date'})
calibrations['date'] = pd.to_datetime(calibrations['date'],format='%Y-%m-%d')
calibrations['risk_free_rate'] = 0.04
calibrations['spread'] = spread
calibrations['n_strikes'] = n_strikes

dates = pd.Series(calibrations['date'].copy().drop_duplicates().tolist())
print(dates)

df_generate_asian_options(calibrations)
