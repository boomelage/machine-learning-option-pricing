import os
import pandas as pd
import numpy as np
from pathlib import Path
from model_settings import ms
script = Path(__file__).resolve().parent.absolute()
data_dir = os.path.join(script.parent.parent.parent,ms.cboe_spx_barrier_dump)
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(data_dir,f)) for f in files]
df = pd.concat(dfs,ignore_index=True).iloc[:,1:].copy()

df['observed_price'] = np.maximum(df['barrier_price'] + np.random.normal(scale=(0.15)**2,size=df.shape[0]),0)
df['observed_price'] = pd.to_numeric(df['observed_price'],errors='coerce')
numerical_features = [
	'spot_price', 'strike_price', 'barrier', 'days_to_maturity',
	'rebate', 'dividend_rate','risk_free_rate', 
	'theta', 'kappa', 'rho', 'eta', 'v0'
	]

categorical_features = ['w', 'barrier_type_name']

feature_set = numerical_features + categorical_features

target_name = 'observed_price'



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = df.dropna().copy()
train_data,test_data = train_test_split(df)
train_X = train_data[feature_set]
train_y = train_data[target_name]
test_X = train_data[feature_set]
test_y = train_data[target_name]


preprocessor = ColumnTransformer(
	[
		("numerical",StandardScaler(),numerical_features),
		("categorical", OneHotEncoder(sparse_output=False),categorical_features)
	]
)


deepnnet_model = MLPRegressor(
	hidden_layer_sizes= (len(feature_set),)*3,
	activation = 'relu', 
	solver= 'lbfgs',
	max_iter = 1000, 
)

dnn_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", deepnnet_model)
        ])

dnn_scaled = TransformedTargetRegressor(
    regressor=dnn_pipeline,
    transformer=Pipeline([
		("StandardScaler", StandardScaler())
	])
)
train_X,train_y = train_X.dropna(),train_y.dropna()


model_fit = dnn_scaled.fit(train_X,train_y)
