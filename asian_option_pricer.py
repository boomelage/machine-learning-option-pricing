import numpy as np
import time
import matplotlib.pyplot as plt
import QuantLib as ql
from scipy.stats import norm
from model_settings import ms
from datetime import datetime
import pandas as pd


def numpy_cartesian_product(*arrays):
    grids = np.meshgrid(*arrays, indexing='ij')
    product = np.stack([grid.ravel() for grid in grids], axis=-1)
    return product


"""

airthmetic asian option


"""
calculation_datetime = datetime.today()
calculation_date = ql.Date(calculation_datetime.day,calculation_datetime.month,calculation_datetime.year)


r = 0.04
g = 0.0
s = 365.41
k = 400
w = 'call'

kappa = 0.412367
theta = 0.17771
rho = -0.582856
eta = 0.785592
v0 = 0.08079 
p = ql.Period("12M")
t = 1
n = 250
m = 10000



dt = t/n

size = (m,n)


riskFreeTS = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, r, ql.Actual365Fixed()))

dividendTS = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, g, ql.Actual365Fixed()))

initialValue = ql.QuoteHandle(ql.SimpleQuote(s))
hestonProcess = ql.HestonProcess(
    riskFreeTS, dividendTS, initialValue, v0, kappa, theta, eta, rho)

dimension = hestonProcess.factors()
times = ql.TimeGrid(t, n)

rng = ql.UniformRandomSequenceGenerator(
    dimension * n, ql.UniformRandomGenerator()
)

sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)

pathGenerator = ql.GaussianMultiPathGenerator(
    hestonProcess, list(times), sequenceGenerator, False)

paths = [[] for i in range(dimension)]
for i in range(m):
    samplePath = pathGenerator.next()
    values = samplePath.value()
    
    for j in range(dimension):
        paths[j].append([x for x in values[j]])

price_paths = np.asarray(paths)[0][:,1:]
S_geo = np.exp(np.mean(np.log(price_paths),axis=1))
S_avg = np.mean(price_paths,axis=1)

discount = np.exp(-r*t)

arithmetic_price = discount * np.mean(np.maximum(S_avg-k,0))
geometric_price = discount * np.mean(np.maximum(S_geo-k,0))
mc_vanilla = discount * np.mean(np.maximum(price_paths[:,-1]-k,0))

t = calculation_date + p - calculation_date 
my_vanilla = ms.ql_heston_price(s,k,t,r,g,w,kappa,theta,rho,eta,v0,calculation_datetime)
print(f"geo: {geometric_price}")
print(f"arith: {arithmetic_price}")
print(f"monte carlo heston vanilla: {mc_vanilla}")
print(f"analytical heston vanilla: {my_vanilla}")
plt.figure()
for path in np.asarray(paths)[0]:
    plt.plot(pd.Series(path))

plt.show()
plt.clf()