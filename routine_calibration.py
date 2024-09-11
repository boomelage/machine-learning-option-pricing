# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:17:45 2024

"""
def clear_all():
    globals_ = globals().copy()  # Make a copy to avoid 
    for name in globals_:        # modifying during iteration
        if not name.startswith('_') and name not in ['clear_all']:
            del globals()[name]
clear_all()
import os
import QuantLib as ql
import warnings
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore')
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

# =============================================================================
                                            # QuantLib pricing settings/objects
dividend_rate = 0.005
risk_free_rate = 0.05
S  = 5400
calculation_date = ql.Date.todaysDate()
day_count = ql.Actual365Fixed()
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(m=1)
ql.Settings.instance().evaluationDate = calculation_date
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
dividend_rate = dividend_yield
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
    calculation_date, dividend_rate, day_count))

# =============================================================================
                                       # creating th implied volatility surface

from ivolmat_from_market import extract_ivol_matrix_from_market

implied_vol_matrix, strikes, maturities, ivoldf = \
    extract_ivol_matrix_from_market(r'SPXts.xlsx')      

expiration_dates = np.empty(len(maturities), dtype=object)
for i, maturity in enumerate(maturities):
    expiration_dates[i] = calculation_date + ql.Period(maturity, ql.Days)

black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vol_matrix, day_count)
import time


# =============================================================================
                                 # Heston model settings and initial parameters

v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;
S = ql.QuoteHandle(ql.SimpleQuote(S))
process = ql.HestonProcess(
    flat_ts, dividend_ts, S, v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)
heston_helpers = []

# =============================================================================
                                                          # calibration routine

for current_index, date in enumerate(expiration_dates):
    print(f"\nCurrently calibrating for maturity: {date}")
    black_var_surface.setInterpolation("bicubic")
    for j, s in enumerate(strikes):
       t = day_count.yearFraction(calculation_date, date)
       sigma = black_var_surface.blackVol(t, s)  
       helper = ql.HestonModelHelper(
           ql.Period(int(t * 365), ql.Days),
           calendar, S.value(), s,
           ql.QuoteHandle(ql.SimpleQuote(sigma)),
           flat_ts, dividend_ts
           )
       helper.setPricingEngine(engine)
       heston_helpers.append(helper)
    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(heston_helpers, lm,
                     ql.EndCriteria(500, 50, 1.0e-8,1.0e-8, 1.0e-8))
    theta, kappa, sigma, rho, v0 = model.params()
    
    print (
        "\ntheta = %f, kappa = %f, "
        "sigma = %f, rho = %f, v0 = %f" % (theta, kappa, 
                                           sigma, rho, v0))
    avg = 0.0
    time.sleep(0.005)
    print ("%15s %15s %15s %20s" % (
        "Strikes", "Market Value",
          "Model Value", "Relative Error (%)"))
    print ("="*70)
    for i in range(min(len(heston_helpers), len(strikes))):
        opt = heston_helpers[i]
        err = (opt.modelValue() / opt.marketValue() - 1.0)
        print(f"{strikes[i]:15.2f} {opt.marketValue():14.5f} "
              f"{opt.modelValue():15.5f} {100.0 * err:20.7f}")
        avg += abs(err)  # accumulate the absolute error
    avg = avg*100.0/len(heston_helpers)
    print("-"*70)
    print("Total Average Abs Error (%%) : %5.3f" % (avg))
    heston_params = {
        'theta':theta, 
        'kappa':kappa, 
        'sigma':sigma, 
        'rho':rho, 
        'v0':v0
        }
    print('\nHeston model parameters:')
    for key, value in heston_params.items():
        print(f'{key}: {value}')

# =============================================================================
                                                                     # plotting
# outputs_path = None
# ticker = None
# from plot_volatility_surface import plot_volatility_surface
# plot_volatility_surface(outputs_path, ticker, ivoldf,strikes,maturities,black_var_surface)