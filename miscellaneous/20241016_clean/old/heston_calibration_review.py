#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:54:40 2024

"""
import QuantLib as ql
import time
def calibrate_heston(flat_ts, dividend_ts, S, expiration_dates,
    black_var_surface, strikes, day_count, calculation_date, calendar,
        dividend_rate):
    # initial guesses
    v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;
    S = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.HestonProcess(
        flat_ts, dividend_ts, S, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    heston_helpers = []
    # loop through all maturities and perform calibration for each one
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
        return heston_params

    