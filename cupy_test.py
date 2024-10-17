import cupy as cp
import numpy as np
import time
from itertools import product
from scipy.stats import norm
import pandas as pd
import QuantLib as ql

def cupy_cartesian_product(*arrays):
    grids = cp.meshgrid(*arrays, indexing='ij')
    product = cp.stack([grid.ravel() for grid in grids], axis=-1)
    return product

def numpy_cartesian_product(*arrays):
    grids = np.meshgrid(*arrays, indexing='ij')
    product = np.stack([grid.ravel() for grid in grids], axis=-1)
    return product

def cupy_black_scholes(s, k, t, r, volatility):
    s = cp.asarray(s)
    k = cp.asarray(k)
    t = cp.asarray(t)
    r = cp.asarray(r)
    volatility = cp.asarray(volatility)
    
    d1 = (cp.log(s / k) + (r + 0.5 * volatility**2) * t) / (volatility * cp.sqrt(t))
    d2 = d1 - volatility * cp.sqrt(t)
    
    N_d1 = norm.cdf(d1.get()) # transfer to numpy
    N_d2 = norm.cdf(d2.get()) # transfer to numpy
    							#	CPU bound!
    N_d1_cp = cp.asarray(N_d1)
    N_d2_cp = cp.asarray(N_d2)
    
    call_price = s * N_d1_cp - k * cp.exp(-r * t) * N_d2_cp

    return call_price

def numpy_black_scholes(s, k, t, r, volatility):
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    r = np.asarray(r)
    volatility = np.asarray(volatility)
    
    d1 = (np.log(s / k) + (r + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
    d2 = d1 - volatility * np.sqrt(t)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    call_price = s * N_d1 - k * np.exp(-r * t) * N_d2

    return call_price


def cupy_geometric_asian_call(s, k, t, r, volatility):
    s = cp.asarray(s)
    k = cp.asarray(k)
    t = cp.asarray(t)
    r = cp.asarray(r)
    volatility = cp.asarray(volatility)
    
    b = r
    
    vol = volatility * cp.sqrt(t / 3)
    d1 = (cp.log(s * cp.exp((b + volatility**2 / 6) * t) / k) + 0.5 * vol**2) / vol
    d2 = d1 - vol
    
    N_d1 = norm.cdf(d1.get())
    N_d2 = norm.cdf(d2.get())
    
    N_d1_cp = cp.asarray(N_d1)
    N_d2_cp = cp.asarray(N_d2)
    
    call_price = cp.exp(-r * t) * (s * cp.exp((b + volatility**2 / 6) * t) * N_d1_cp - k * N_d2_cp)
    
    return call_price


def numpy_geometric_asian_call(s, k, t, r, volatility):
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    r = np.asarray(r)
    volatility = np.asarray(volatility)
    
    b = r
    
    vol = volatility * np.sqrt(t / 3)
    d1 = (np.log(s * np.exp((b + volatility**2 / 6) * t) / k) + 0.5 * vol**2) / vol
    d2 = d1 - vol
    
    N_d1 = norm.cdf(d1) 
    N_d2 = norm.cdf(d2)
    
    call_price = np.exp(-r * t) * (s * np.exp((b + volatility**2 / 6) * T) * N_d1 - k * N_d2)
    
    return call_price



s = 100
k = 120
t = 1
r = 0.04
volatility = 0.4

S = np.linspace(90,110,10000)
K = np.linspace(80,120,1000)
T = [t]
R = [r]
vols = [volatility]

S_cp = cp.asarray(S)
K_cp = cp.asarray(K)
T_cp = cp.asarray(T)
R_cp = cp.asarray(R)
vols_cp = cp.asarray(vols)


"""
numpy
"""

start = time.time()
features = numpy_cartesian_product(S, K, T, R, vols)
features = features.transpose()
S, K, T, R, vols = features[0], features[1], features[2], features[3], features[4]

black_scholes = numpy_black_scholes(S,K,T,R,vols)
geometric_asian_options = numpy_geometric_asian_call(S,K,T,R,vols)

end = time.time()
df = pd.DataFrame(
	{
	"spot_price":S,
	"strike_price":K,
	"days_to_maturity":T,
	"risk_free_rate":R,
	"volatility":vols,
	"black_scholes":black_scholes,
	"geometric_asian":geometric_asian_options
	}
)

print(df)
print('numpy time:')
print(end-start)



"""
cupy
"""
start = time.time()
features_cp = cupy_cartesian_product(S_cp, K_cp, T_cp, R_cp, vols_cp)
features_cp = features_cp.transpose()
S_cp, K_cp, T_cp, R_cp, vols_cp = features_cp[0], features_cp[1], features_cp[2], features_cp[3], features_cp[4]

black_scholes = cupy_black_scholes(S_cp,K_cp,T_cp,R_cp,vols_cp)
geometric_asian_options = cupy_geometric_asian_call(S_cp,K_cp,T_cp,R_cp,vols_cp)
end = time.time()

df = pd.DataFrame(
	{
	"spot_price":S_cp.get(),
	"strike_price":K_cp.get(),
	"days_to_maturity":T_cp.get(),
	"risk_free_rate":R_cp.get(),
	"volatility":vols_cp.get(),
	"black_scholes":black_scholes.get(),
	"geometric_asian_price":geometric_asian_options.get(),
	}
)


print(df)
print('cupy time:')
print(end-start)



