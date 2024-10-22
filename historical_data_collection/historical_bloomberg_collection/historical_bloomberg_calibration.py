import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.getcwd(),'surface_modelling'))
from Derman import derman
from historical_collection import historical_data
from historical_bloomberg_surface_modelling import derman_coefs
from heston_model_calibration import calibrate_heston

historical_data = historical_data.iloc[:100].copy()
historical_data.loc[:,['kappa','theta','rho','eta','v0']] = np.tile(np.nan,(historical_data.shape[0],5))
for i,row in historical_data.iterrows():
	calculation_datetime = row['date']
	s = float(row['spot_price'])
	g = float(row['dividend_rate'])
	r = 0.04
	T = [31,60,196,95,368] # hard coded due to inflexibility of method and data availability contraints
	atm_vols = row[['30D', '60D', '3M', '6M', '12M']].squeeze()
	atm_vols.index = T

	if calculation_datetime == pd.Timestamp(2008,11,20):
		spread = 0.30
	elif s < 900:
		spread = 0.25
	elif s < 1000:
		spread = 0.20
	elif s < 1100:
		spread = 0.15
	elif s < 1200:
		spread = 0.10
	else:
		spread = 0.05

	K = np.linspace(s*(1-spread),s*(1+spread),5,dtype=int)

	vol_matrix = pd.DataFrame(np.tile(np.nan,(len(K),len(T))),index=K,columns=T)


	for k in K:
		for t in T:
			vol_matrix.loc[k,t] = atm_vols.loc[t] + derman_coefs.loc[t]*(k-s)

	heston_parameters = pd.Series(calibrate_heston(vol_matrix,s,r,g))
	for param, value in heston_parameters.items():
		historical_data.loc[i,param] = value
	print(f"\n{heston_parameters}")			

print(f"\n{historical_data}")