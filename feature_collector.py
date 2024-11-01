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
	dataset[price_name] = pd.to_numeric(dataset[price_name],errors='coerce')
	dataset = dataset[dataset[price_name]<=dataset['spot_price']]
	dataset['observed_price'] = np.maximum(dataset[price_name] + np.random.normal(scale=(0.15)**2,size=dataset.shape[0]),0)
	dataset = dataset[dataset['observed_price']>=0.01]
	dataset['calculation_date'] = pd.to_datetime(dataset['calculation_date'])
	dataset = dataset.sort_values(by='calculation_date',ascending=False).reset_index(drop=True)
	return dataset


def preprocess_data(dataset,development_dates,test_dates,trainer):
    train_data = dataset[dataset['calculation_date'].isin(development_dates)]
    test_data = dataset[dataset['calculation_date'].isin(test_dates)]
    
    train_plot = train_data[['calculation_date','spot_price']].copy().set_index('calculation_date').drop_duplicates()
    test_plot = test_data[['calculation_date','spot_price']].copy().set_index('calculation_date').drop_duplicates()
    
    trainplotx = pd.date_range(start=min(train_plot.index),end=max(train_plot.index),periods=len(train_plot))
    testplotx = pd.date_range(start=min(test_plot.index),end=max(test_plot.index),periods=len(test_plot))
    
    plt.figure()
    plt.xticks(rotation=45)
    plt.plot(testplotx,test_plot,color='purple',label='out-of-sample')
    plt.plot(trainplotx,train_plot,color='green',label='in-sample')
    plt.legend()
    plt.show()
    arrs = trainer.get_train_test_arrays(
        train_data, test_data)
    train_X = arrs['train_X']
    train_y = arrs['train_y']
    test_X = arrs['test_X']
    test_y = arrs['test_y']
    preprocessor = trainer.preprocess()
    print(len(train_y),len(train_X))
    return {'preprocessor':preprocessor,'train_test_arrays':arrs}