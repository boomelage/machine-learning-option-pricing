from scipy.stats import qmc
import pandas as pd
pd.set_option('display.max_columns',None)
sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=int(1e5))


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

sample_scaled['feller'] = 2*sample_scaled['kappa']*sample_scaled['theta']-sample_scaled['eta']**2
sample_scaled = sample_scaled[abs(sample_scaled)<1].dropna().reset_index(drop=True)
print(sample_scaled.describe())
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(sample_scaled)
plt.show()

from pathlib import Path
from model_settings import ms
ms.find_root(Path())
ms.collect_spx_calibrations()
market = ms.spx_calibrations[sample_scaled.columns]
# market = market[abs(market)<1].dropna().reset_index(drop=True)

print(market.describe())
sns.pairplot(market)
plt.show()