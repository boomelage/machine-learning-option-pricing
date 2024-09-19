# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:13:18 2024

"""
import os
import sys
import time
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')

train_start = time.time()
train_start_datetime = datetime.fromtimestamp(train_start)
train_start_tag = train_start_datetime.strftime('%c')
print(f"\n{train_start_tag}\n")

from train_generation import ml_data

from mlop import mlop
mlop = mlop(user_dataset=ml_data)

train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.split_user_data()

preprocessor = mlop.preprocess()


# model_fit, runtime = mlop.run_nnet(preprocessor, train_X, train_y)

# model_fit, runtime = mlop.run_dnn(preprocessor,train_X,train_y)

model_fit = mlop.run_rf(preprocessor,train_X,train_y)

# model_fit = mlop.run_lm(train_X,train_y)


df = mlop.compute_predictive_performance(test_data,test_X,model_fit)

predictive_performance_plot = mlop.plot_model_performance(df)

train_end = time.time()

train_time = train_end - train_start

print(f"\nruntime: {int(train_time)} seconds")

