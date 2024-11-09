from scipy.stats import qmc
import pandas as pd
pd.set_option('display.max_columns',None)
sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=100000)


l_bounds = [
	0.000125, 
	0.01, 
	-1, 
	0.05, 
	0.004, 
]


u_bounds = [
	10, 
	0.80, 
	-0.2, 
	0.80, 
	0.2, 
]


sample_scaled = pd.DataFrame(
	qmc.scale(sample, l_bounds, u_bounds),
	columns=['kappa','theta','rho','eta','v0']
)



print(sample_scaled.describe())