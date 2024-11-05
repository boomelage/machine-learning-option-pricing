import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime



from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline


def make_dnn_pipeline(feature_set,numerical_scaler):
    preprocessor = ColumnTransformer([
    	('scaling',numerical_scaler,feature_set[:-2]),
    	('categorical',OneHotEncoder(),['w','barrier_type_name'])
	])
    dnn_pipeline = make_pipeline(preprocessor,MLPRegressor(max_iter=1000,random_state=1312))
    return TransformedTargetRegressor(
        regressor=dnn_pipeline,
        transformer= Pipeline([("StandardScaler", StandardScaler())])
    )


numerical_scaler = StandardScaler()
price_name = 'barrier_price'

from model_settings import ms
root = Path().resolve().parent.parent
data_dir = os.path.join(root,ms.cboe_spx_barriers['dump'])
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
dates = pd.Series([d[:d.find('_',0)] for d in files]).drop_duplicates().reset_index(drop=True)



# for date in dates:
# date_files = [f for f in files if f.find(date)!=-1]

file_paths = [os.path.join(data_dir,f) for f in files]
dfs = [pd.read_csv(f).dropna() for f in file_paths]

if len(dfs)>0:
	dataset = pd.concat(dfs,ignore_index=True).iloc[:,1:]


	dataset = dataset[dataset[price_name]<dataset['spot_price']].dropna().reset_index(drop=True).copy()



	dataset['observed_price'] = np.maximum(dataset[price_name] + np.random.normal(scale=(0.15)**2,size=dataset.shape[0]),0)
	dataset['moneyness'] = ms.df_moneyness(dataset)
	dataset['relative_moneyness'] = dataset['moneyness']/dataset['spot_price']
	dataset['relative_barrier'] = dataset['barrier']/dataset['spot_price']
	dataset['relative_price'] = dataset['observed_price']/dataset['spot_price']
	# dataset = dataset[(dataset['w']=='put')&(dataset['barrier_type_name']=='DownOut')].reset_index(drop=True)



	spot_dates = pd.Series(np.sort(dataset['date'].unique()))
	development_dates = spot_dates[:200]
	test_dates = spot_dates[~spot_dates.isin(development_dates)]
	train_data = dataset[dataset['date'].isin(development_dates)]
	test_data = dataset[dataset['date'].isin(development_dates)]

	feature_set = [
	    'relative_moneyness',
	    'days_to_maturity',
	    'risk_free_rate',
	    'dividend_rate',
	    'kappa',
	    'theta',
	    'rho',
	    'eta',
	    'v0',
	    'relative_barrier',
	    'barrier_type_name',
	    'w'
	]

	train_X = train_data[feature_set]
	train_y = train_data['relative_price']

	dnn = make_dnn_pipeline(feature_set,numerical_scaler)

	model_fit = dnn.fit(train_X,train_y)
	train_pred = model_fit.predict(train_X)*train_data['spot_price']
	print('Oosterlee MAE:',np.mean(np.abs(train_pred-train_data['spot_price'])))



	feature_set = [
	    'spot_price',
	    'strike_price',
	    'days_to_maturity',
	    'risk_free_rate',
	    'dividend_rate',
	    'kappa',
	    'theta',
	    'rho',
	    'eta',
	    'v0',
	    'barrier',
	    'barrier_type_name',
	    'w'
	]


	train_X = train_data[feature_set]
	train_y = train_data['observed_price']

	dnn = make_dnn_pipeline(feature_set,numerical_scaler)

	model_fit = dnn.fit(train_X,train_y)
	train_pred = model_fit.predict(train_X)
	print('equivalent unparameterized MAE:',np.mean(np.abs(train_pred-train_data['spot_price'])))

	from convsklearn import barrier_trainer
	barrier_trainer.dnn_params['random_state'] = 1312
	barrier_trainer.feature_set = feature_set
	arrs = barrier_trainer.get_train_test_arrays(train_data,test_data)
	preprocessor = barrier_trainer.preprocess()
	old_dnn = barrier_trainer.run_dnn(preprocessor,train_X,train_y,print_details=False)
	train_pred = old_dnn.predict(train_X)
	print('my MAE:',np.mean(np.abs(train_pred-train_data['spot_price'])))