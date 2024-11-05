import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_data(dataset,development_dates,test_dates,trainer):
    try:
        train_data = dataset[dataset['date'].isin(development_dates)].sort_values(by='date')
        test_data = dataset[dataset['date'].isin(test_dates)].sort_values(by='date')
    except Exception:
        train_data = dataset[dataset['calculation_date'].isin(development_dates)].sort_values(by='calculation_date')
        test_data = dataset[dataset['calculation_date'].isin(test_dates)].sort_values(by='calculation_date')

    trainplotx = pd.date_range(start=min(development_dates),end=max(development_dates),periods=train_data.shape[0])
    testplotx = pd.date_range(start=min(test_dates),end=max(test_dates),periods=test_data.shape[0])

    plt.figure()
    plt.plot(testplotx,test_data['spot_price'].values,color='purple',label='out-of-sample')
    plt.plot(trainplotx,train_data['spot_price'].values,color='green',label='in-sample')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.show()
    arrs = trainer.get_train_test_arrays(
    train_data, test_data)
    train_X = arrs['train_X']
    train_y = arrs['train_y']
    test_X = arrs['test_X']
    test_y = arrs['test_y']
    preprocessor = trainer.preprocess()
    return {'preprocessor':preprocessor,'train_test_arrays':arrs,'train_data':train_data,'test_data':test_data}