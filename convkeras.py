import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class convkeras:
	def __init__(self):
		self.train_X = {}
		self.train_y = {}
		self.layers = []
		self.sgd_params = {'learning_rate':0.01}
		self.loss='mean_squared_error'

	def adapt_scaler(self):
		scaler = tf.keras.layers.Normalization(axis=-1)
		scaler.adapt(np.array(self.test_X))
		self.scaler = scaler

	def specify_model(self,layers=None):
		if layers == None:
			layers = self.scaler + self.layers
		model = keras.Sequential(layers)
		model.compile(
			loss=self.loss,optimizer=tf.keras.optimizers.SGD(**self.sgd_params)
		)
		self.model = model

	def fit_model(self,epochs,verbose=1,validation_split=0.05):
		self.history = self.model.fit(
			self.train_X,self.train_y,
			verbose=verbose,validation_split=0.05,
			epochs=epochs
		)


# ck = convkeras()
# layers = [
# 	layers.Dense(13, activation='relu'),
# 	layers.Dense(13, activation='relu'),
# 	layers.Dense(13, activation='relu'),
# 	layers.Dense(13, activation='relu'),
# 	layers.Dense(13, activation='relu'),
# 	layers.Dense(1, activation='linear')
# ]
# from testing_keras import train_X, train_y, test_X, train_y
# ck.train_X = train_X
# ck.specify_model(layers)
# ck.layers = layers
# ck.train_y = train_y
# ck.fit_model(epochs=10)