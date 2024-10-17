import cupy as cp
import numpy as np
import time
from itertools import product
from scipy.stats import norm

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


s = 100
k = 120
t = 1
r = 0.04
volatility = 0.4

S = np.linspace(90,110,10000)
K = np.linspace(80,120,5000)
T = [t]
R = [r]
vols = [volatility]

features = np.array(list(product(S,K,T,R,vols))).transpose()

S,K,T,R,vols = features[0],features[1],features[2],features[3],features[4]



cupy_start = time.time()
calls = cupy_black_scholes(S,K,T,R,vols)
cupy_end = time.time()
print(f"cupy time: {cupy_end - cupy_start}")

numpy_start = time.time()
numpy_calls = numpy_black_scholes(S,K,T,R,vols)
numpy_end = time.time()
print(f"numpy time: {numpy_end - numpy_start}")




def cupy_geometric_asian_call(S0, K, T, r, sigma):
    # Ensure inputs are CuPy arrays
    S0 = cp.asarray(S0)
    K = cp.asarray(K)
    T = cp.asarray(T)
    r = cp.asarray(r)
    sigma = cp.asarray(sigma)
    
    b = r
    
    # Compute d1 and d2 for geometric Asian option
    sigma_adj = sigma * cp.sqrt(T / 3)  # Adjusted volatility for geometric Asian options
    d1 = (cp.log(S0 * cp.exp((b + sigma**2 / 6) * T) / K) + 0.5 * sigma_adj**2) / sigma_adj
    d2 = d1 - sigma_adj
    
    # Use SciPy's norm.cdf to compute the cumulative normal distribution (transfer to NumPy for norm.cdf)
    N_d1 = norm.cdf(d1.get())  # Transfer to NumPy for norm.cdf
    N_d2 = norm.cdf(d2.get())  # Transfer to NumPy for norm.cdf
    
    # Convert back to CuPy arrays
    N_d1_cp = cp.asarray(N_d1)
    N_d2_cp = cp.asarray(N_d2)
    
    # Compute the price of the geometric Asian call option
    call_price = cp.exp(-r * T) * (S0 * cp.exp((b + sigma**2 / 6) * T) * N_d1_cp - K * N_d2_cp)
    
    return call_price

geometric_asian_call_price = cupy_geometric_asian_call(S, K, T, R, vols)

print("Geometric Asian Call Option Price:", geometric_asian_call_price.get())
