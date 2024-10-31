import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_features(datadir,price_name):
	files = [f for f in os.listdir(datadir) if f.endswith('.csv')]
	files = [os.path.join(datadir,f) for f in files]
	dfs = []
	bar = tqdm(total=len(files)+1)
	for f in files:
	    dfs.append(pd.read_csv(f).iloc[:,1:])
	    bar.update(1)
	dataset = pd.concat(dfs,ignore_index=True).dropna().reset_index(drop=True)
	bar.update(1)
	bar.close()
	dataset['calculation_date'] = pd.to_datetime(dataset['calculation_date'],format='mixed')
	dataset[price_name] = pd.to_numeric(dataset[price_name],errors='coerce')
	dataset = dataset[dataset[price_name]<=dataset['spot_price']]
	dataset['observed_price'] = np.maximum(dataset[price_name] + np.random.normal(scale=(0.15)**2,size=dataset.shape[0]),0)
	dataset = dataset[dataset['observed_price']>=0.01]
	dataset['calculation_date'] = pd.to_datetime(dataset['calculation_date'])
	dataset = dataset.sort_values(by='calculation_date').reset_index(drop=True)
	return dataset