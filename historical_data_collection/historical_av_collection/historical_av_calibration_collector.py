import os
import sys
import pandas as pd
import numpy as np

files_dir = os.path.join(os.getcwd(),'av_calibrations')
os.chdir(files_dir)
files = [f for f in os.listdir() if f.endswith('.csv')]

dfs = []

for f in files:
	dfs.append(pd.read_csv(f))

calibrations = pd.concat(dfs,ignore_index=True).iloc[:,1:]

show = calibrations.copy()
show['difference'] = show['heston']-show['black_scholes']
error = show['difference']
RMSE = np.sqrt(np.mean(error**2))
MAE = np.mean(np.abs(error))
avg = np.mean(np.abs(show['relative_error']))
print(f"RMSE {RMSE}", f"MAE: {MAE}", f"mean absolute relative error: {round(100*avg,4)}%")

print(calibrations.columns)
calibrations = calibrations[
	[
		'calculation_date','spot_price','risk_free_rate', 'dividend_rate', 
		'kappa', 'theta', 'rho', 'eta', 'v0'
	]
].drop_duplicates().copy()

print(calibrations)







