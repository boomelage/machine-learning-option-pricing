"""
https://www.tensorflow.org/tutorials/keras/regression
https://www.tensorflow.org/guide/keras/preprocessing_layers


sns.pairplot(train_dataset[['relative_spot','kappa', 'theta', 'rho', 'eta', 'v0',]], diag_kind='kde')
"""

import os
from time import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from df_collector import df_collector
from model_settings import ms
from convsklearn import convsklearn

np.set_printoptions(precision=3, suppress=True)
ms.find_root(Path())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tic = time()

"""

loading data

"""
datadir = os.path.join(ms.root,ms.cboe_spx_short_term_asians['dump'])
df = df_collector.collect_dfs(datadir).iloc[:,1:]

conv = convsklearn()
conv.load_data(df)

if datadir.find('asian') != -1:
    pricename = 'asian_price'

elif datadir.find('barrier') !=-1:
    pricename = 'barrier_price'
else:
    print('unknown model')
    pricename = ''

df['relative_price'] = df[pricename]/df['strike_price']
df['relative_spot'] = df['spot_price']/df['strike_price']

cats = conv.categorical_features
nums = conv.numerical_features
nums = [n for n in nums if n.find('strike_price')==-1 and n.find('spot_price')==-1]
nums = nums + ['relative_spot']




"""
preprocessing
"""

dataset = df.copy()
dataset = dataset[dataset]
dataset['calculation_date'] = pd.to_datetime(dataset['calculation_date'],format='mixed')
dataset['date'] = dataset['calculation_date'].dt.floor('D')
dates = dataset['date'].drop_duplicates().reset_index(drop=True)
dataset.tail()
dataset = dataset.dropna()
dataset = pd.get_dummies(dataset, columns=['w','averaging_type'], prefix='', prefix_sep='')

development_dates = dates.iloc[:100]
test_dates = dates[~dates.isin(development_dates)]
train_data = dataset[dataset['date'].isin(development_dates)].copy()
test_data = dataset[dataset['date'].isin(test_dates)].copy()

target = 'relative_price'
train_X = train_data[nums+train_data.columns.tolist()[-4:]]
test_X = test_data[nums+test_data.columns.tolist()[-4:]]

train_y = train_data[target]
test_y = test_data[target]


"""
mlp
"""


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(test_X))

n_features = len(cats)+len(nums)

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(n_features, activation='relu'),
        layers.Dense(n_features, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
    )
    return model

mlp = build_and_compile_model(normalizer)

history = mlp.fit(
    train_X,
    train_y,
    validation_split=0.05,
    verbose=1, epochs=500)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
loss = hist.iloc[:,:-1]

test_results = {}
test_results['mlp'] = mlp.evaluate(test_X, test_y, verbose=0)

insample_prediction = mlp.predict(train_X).flatten()
insample_error = train_y-insample_prediction

plt.figure()
plt.hist(insample_error,bins = 50)
plt.show()
plt.figure()
plt.plot(loss)
plt.show()
print(train_X.describe().T)
print(insample_error.describe())

tictoc = time()-tic
