import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    try:
        dataset['date'] = pd.to_datetime(dataset['date'],format='mixed')
    except Exception:
        pass
    dataset[price_name] = pd.to_numeric(dataset[price_name],errors='coerce')
    dataset = dataset[dataset[price_name]<=dataset['spot_price']]
    dataset['observed_price'] = np.maximum(dataset[price_name] + np.random.normal(scale=(0.15)**2,size=dataset.shape[0]),0)
    dataset = dataset[dataset['observed_price']>=0.01]
    dataset = dataset.sort_values(by='calculation_date',ascending=False).reset_index(drop=True)
    return dataset


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
    plt.legend()
    plt.show()
    arrs = trainer.get_train_test_arrays(
    train_data, test_data)
    train_X = arrs['train_X']
    train_y = arrs['train_y']
    test_X = arrs['test_X']
    test_y = arrs['test_y']
    preprocessor = trainer.preprocess()
    return {'preprocessor':preprocessor,'train_test_arrays':arrs,'train_data':train_data,'test_data':test_data}

