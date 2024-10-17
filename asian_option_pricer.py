import numpy as np
import time
import matplotlib.pyplot as plt
import QuantLib as ql
import pandas as pd
from scipy.stats import norm
from model_settings import ms
from datetime import datetime


def numpy_cartesian_product(*arrays):
    grids = np.meshgrid(*arrays, indexing='ij')
    product = np.stack([grid.ravel() for grid in grids], axis=-1)
    return product


"""

airthmetic asian option


"""

r = 0.04
g = 0.0
s = 365.41
k = 200
w = 'call'

kappa = 0.412367
theta = 0.17771
rho = -0.582856
eta = 0.785592
v0 = 0.08079 
t = 2

n = 1
m = 10000
dt = t/n


size = (m,n)


today = ql.Date().todaysDate()

riskFreeTS = ql.YieldTermStructureHandle(
    ql.FlatForward(today, r, ql.Actual365Fixed()))

dividendTS = ql.YieldTermStructureHandle(
    ql.FlatForward(today, g, ql.Actual365Fixed()))

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
S_t = np.asarray(price_paths)[:,-1]
S_avg = np.mean(price_paths,axis=1)
S_geo = np.prod(price_paths,axis=1)**(1/n)


MC_vanilla = np.exp(-r*t)*np.mean(np.maximum(S_t-k,0))
arithmetic_asian = np.exp(-r*t)*np.mean(np.maximum(S_avg-k,0))
geometric_asian = np.exp(-r*t)*np.mean(np.maximum(S_geo-k,0))
# heston_vanilla = ms.ql_heston_price(s,2,k,r,g,w,kappa,theta,rho,eta,v0,datetime.today())

print(f"geo: {geometric_asian}")
print(f"arith: {arithmetic_asian}")
print(f"monte carlo heston vanilla: {MC_vanilla}")
# print(f"analytic heston vanilla: {heston_vanilla}")

plt.figure()
for path in np.asarray(paths)[0]:
    plt.plot(pd.Series(path))

plt.show()
plt.clf()