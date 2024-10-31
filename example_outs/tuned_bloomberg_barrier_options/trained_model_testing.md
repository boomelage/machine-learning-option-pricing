```python
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

# functions


```python
def compute_RMSE(diff):
    if len(diff)>0:
        return np.sqrt(np.mean(diff.values**2))
        
def compute_MAE(diff):
    if len(diff)>0:
        return np.mean(np.abs(diff.values))

def plot_errors(plotcols, test_data, train_data):
    test_diff = test_data['outofsample_error']
    train_diff = train_data['insample_error']
    
    test_data['RMSE'] = test_diff.resample('D').apply(compute_RMSE).dropna()
    test_data['MAE'] = test_diff.resample('D').apply(compute_MAE).dropna()
    test_plot = test_data[plotcols].copy().drop_duplicates()
    
    train_data['RMSE'] = train_diff.resample('D').apply(compute_RMSE).dropna()
    train_data['MAE'] = train_diff.resample('D').apply(compute_MAE).dropna()
    train_plot = train_data[plotcols].copy().drop_duplicates()
    
    fig,axs = plt.subplots(len(plotcols),figsize=(10,10),sharex=True)
    for i,col in enumerate(plotcols):
        axs[i].plot(train_plot[col],color='green',label='in-sample')
        axs[i].set_title(col.replace('_',' '))
        axs[i].legend()
    for i,col in enumerate(plotcols):
        axs[i].plot(test_plot[col],color='purple',label='out-of-sample')
        axs[i].set_title(col.replace('_',' '))
        axs[i].legend()
    plt.show()

def retrain(old_train_data,old_test_data,train_to_date):
    new_train = pd.concat([old_train_data,old_test_data[old_test_data.index<=train_to_date]],ignore_index=False).dropna(how='any',axis=1).reset_index(drop=False)
    new_test = old_test_data[~old_test_data.index.isin(new_train['calculation_date'])].copy().reset_index(drop=False)
    arrs = trainer.get_train_test_arrays(new_train, new_test)
    train_X = arrs['train_X']
    train_y = arrs['train_y']
    test_X = arrs['test_X']
    test_y = arrs['test_y']
    preprocessor = trainer.preprocess()
    retrained_model = trainer.run_dnn(preprocessor,train_X,train_y)
    train_test = trainer.test_prediction_accuracy(new_train,new_test,retrained_model)
    new_test_data = train_test['test_data'].set_index('calculation_date')
    new_train_data = train_test['train_data'].set_index('calculation_date')
    print()
    plot_errors(plotcols,new_test_data,new_train_data)
    return new_train_data, new_test_data
```

# loading model


```python
from model_settings import ms
root = Path().resolve().parent.parent
models_dir = os.path.join(root,ms.trained_models)
models = [f for f in os.listdir(models_dir) if f.find('ipynb')==-1]
for i,m in enumerate(models):
    print(f"{i}     {m}")
```

    0     2024_10_31 203810934918 tuned bloomberg barriers
    1     2024_10_31 204451887463 tuned bloomberg asians
    

# loading data


```python
"""
select model here
"""
model = models[0]
""""""

plotcols = ['v0','RMSE', 'MAE','spot_price']

from convsklearn import asian_trainer, barrier_trainer

model_dir = os.path.join(models_dir,model)
model_files = [f for f in os.listdir(model_dir) if f.find('ipynb')==-1 and f.find('.html')==-1]
for i,m in enumerate(model_files):
    print(f"{i}     {m}")
print()
if any('asian' in file for file in model_files):
    trainer = asian_trainer
if any('barrier' in file for file in model_files):
    trainer = barrier_trainer

train_data = pd.read_csv(os.path.join(model_dir,[f for f in model_files if f.find('train')!=-1][0])).iloc[:,1:].copy()
test_data = pd.read_csv(os.path.join(model_dir,[f for f in model_files if f. find('test')!=-1][0])).iloc[:,1:].copy()
train_data['calculation_date'] = pd.to_datetime(train_data['calculation_date'])
test_data['calculation_date'] = pd.to_datetime(test_data['calculation_date'])
train_data = train_data.set_index('calculation_date')
test_data = test_data.set_index('calculation_date')
test_dates = pd.Series(test_data.index).sort_values(ascending=True).drop_duplicates().reset_index(drop=True)
model_fit = joblib.load(os.path.join(model_dir,[f for f in model_files if f.endswith('.pkl')][0]))

for col in trainer.feature_set:
    print(f"{col.replace("_"," "):}",f"\n{test_data[col].copy().squeeze().sort_values().drop_duplicates().reset_index(drop=True)}\n")
print()
print(model_fit)

plot_errors(plotcols, test_data, train_data)
```

    0     2024_10_31 203810934918 tuned bloomberg barriers test_data.csv
    1     2024_10_31 203810934918 tuned bloomberg barriers train_data.csv
    2     2024_10_31 203810934918 tuned bloomberg barriers.pkl
    
    spot price 
    0        676.03
    1        682.43
    2        683.09
    3        696.44
    4        701.13
             ...   
    1095    1461.17
    1096    1461.21
    1097    1461.31
    1098    1461.36
    1099    1465.27
    Name: spot_price, Length: 1100, dtype: float64
    
    strike price 
    0        608.42700
    1        614.18700
    2        614.78100
    3        625.32775
    4        626.79600
               ...    
    9885    1607.28700
    9886    1607.33100
    9887    1607.44100
    9888    1607.49600
    9889    1611.79700
    Name: strike_price, Length: 9890, dtype: float64
    
    days to maturity 
    0     60
    1     90
    2    180
    3    360
    4    540
    5    720
    Name: days_to_maturity, dtype: int64
    
    risk free rate 
    0    0.04
    Name: risk_free_rate, dtype: float64
    
    dividend rate 
    0       0.017912
    1       0.017942
    2       0.017970
    3       0.017981
    4       0.017985
              ...   
    1008    0.035255
    1009    0.035276
    1010    0.035596
    1011    0.035690
    1012    0.037735
    Name: dividend_rate, Length: 1013, dtype: float64
    
    kappa 
    0        0.083258
    1        0.090921
    2        0.098043
    3        0.098494
    4        0.102884
              ...    
    1106    12.960913
    1107    13.113228
    1108    13.250228
    1109    15.696333
    1110    15.991443
    Name: kappa, Length: 1111, dtype: float64
    
    theta 
    0       0.037838
    1       0.038747
    2       0.039401
    3       0.040521
    4       0.040844
              ...   
    1106    0.311501
    1107    0.312746
    1108    0.340959
    1109    0.350397
    1110    0.356100
    Name: theta, Length: 1111, dtype: float64
    
    rho 
    0      -1.000000
    1      -1.000000
    2      -1.000000
    3      -1.000000
    4      -1.000000
              ...   
    1106   -0.234005
    1107   -0.220616
    1108   -0.203652
    1109   -0.196170
    1110   -0.181764
    Name: rho, Length: 1111, dtype: float64
    
    eta 
    0       0.110964
    1       0.112310
    2       0.113278
    3       0.113457
    4       0.113904
              ...   
    1106    1.109709
    1107    1.213266
    1108    1.394869
    1109    1.523884
    1110    1.528820
    Name: eta, Length: 1111, dtype: float64
    
    v0 
    0       0.007385
    1       0.007560
    2       0.008220
    3       0.008839
    4       0.009003
              ...   
    1106    0.527421
    1107    0.620853
    1108    0.631678
    1109    0.652337
    1110    0.694016
    Name: v0, Length: 1111, dtype: float64
    
    barrier 
    0         338.015
    1         341.215
    2         341.545
    3         348.220
    4         350.565
               ...   
    10994    2191.755
    10995    2191.815
    10996    2191.965
    10997    2192.040
    10998    2197.905
    Name: barrier, Length: 10999, dtype: float64
    
    barrier type name 
    0     DownIn
    1    DownOut
    2       UpIn
    3      UpOut
    Name: barrier_type_name, dtype: object
    
    w 
    0    call
    1     put
    Name: w, dtype: object
    
    
    TransformedTargetRegressor(regressor=Pipeline(steps=[('preprocessor',
                                                          ColumnTransformer(transformers=[('StandardScaler',
                                                                                           StandardScaler(),
                                                                                           ['spot_price',
                                                                                            'strike_price',
                                                                                            'days_to_maturity',
                                                                                            'risk_free_rate',
                                                                                            'dividend_rate',
                                                                                            'kappa',
                                                                                            'theta',
                                                                                            'rho',
                                                                                            'eta',
                                                                                            'v0',
                                                                                            'barrier']),
                                                                                          ('OneHotEncoder',
                                                                                           OneHotEncoder(sparse_output=False),
                                                                                           ['barrier_type_name',
                                                                                            'w'])])),
                                                         ('regressor',
                                                          MLPRegressor(alpha=0.01,
                                                                       hidden_layer_sizes=(13,
                                                                                           13),
                                                                       learning_rate='adaptive',
                                                                       learning_rate_init=0.1,
                                                                       max_iter=500,
                                                                       solver='sgd',
                                                                       warm_start=True))]),
                               transformer=Pipeline(steps=[('StandardScaler',
                                                            StandardScaler())]))
    


    
![png](output_6_1.png)
    



```python
from sklearn.inspection import partial_dependence,PartialDependenceDisplay

part_disp_X = train_data[model_fit.feature_names_in_]
fig, ax = plt.subplots(figsize=(12, 6))
disp = PartialDependenceDisplay.from_estimator(model_fit, part_disp_X, ['spot_price','v0'], ax=ax)
```


    
![png](output_7_0.png)
    


# retraining


```python
retraining_frequency = 100  # days
retraining_i = np.arange(retraining_frequency,len(test_dates),retraining_frequency)
retraining_dates = test_dates[retraining_i].reset_index(drop=True)
print(retraining_dates)
```

    0    2008-12-24
    1    2009-05-20
    2    2009-10-12
    3    2010-03-08
    4    2010-07-29
    5    2010-12-20
    6    2011-05-13
    7    2011-10-05
    8    2012-02-29
    9    2012-07-23
    10   2012-12-14
    Name: calculation_date, dtype: datetime64[ns]
    


```python
for date in retraining_dates:
    print()
    print(date.strftime('%c'))
    retrain(train_data,test_data,date)
    print()
```

    
    Wed Dec 24 00:00:00 2008
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 188.96502041816711
    
    in sample:
         RMSE: 6.925477806667283
         MAE: 4.75289190578037
    
    out of sample:
         RMSE: 10.25999720041812
         MAE: 7.162937871032691
    
    


    
![png](output_10_1.png)
    


    
    
    Wed May 20 00:00:00 2009
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 219.64569759368896
    
    in sample:
         RMSE: 6.401899233756155
         MAE: 4.284331535648723
    
    out of sample:
         RMSE: 7.925224685896037
         MAE: 5.344020898004696
    
    


    
![png](output_10_3.png)
    


    
    
    Mon Oct 12 00:00:00 2009
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 250.89609050750732
    
    in sample:
         RMSE: 6.159676757850145
         MAE: 4.151175478612135
    
    out of sample:
         RMSE: 6.971444945265126
         MAE: 4.663655434129119
    
    


    
![png](output_10_5.png)
    


    
    
    Mon Mar  8 00:00:00 2010
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 279.69111466407776
    
    in sample:
         RMSE: 7.032231107899734
         MAE: 4.709252412774634
    
    out of sample:
         RMSE: 8.178374754983283
         MAE: 5.524399800805546
    
    


    
![png](output_10_7.png)
    


    
    
    Thu Jul 29 00:00:00 2010
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 360.31453037261963
    
    in sample:
         RMSE: 6.162064366557745
         MAE: 4.106827823890553
    
    out of sample:
         RMSE: 7.59109145053949
         MAE: 5.076467726930883
    
    


    
![png](output_10_9.png)
    


    
    
    Mon Dec 20 00:00:00 2010
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 378.21457719802856
    
    in sample:
         RMSE: 5.4925520731694295
         MAE: 3.6659259321216835
    
    out of sample:
         RMSE: 7.182671309435807
         MAE: 4.7971283981828226
    
    


    
![png](output_10_11.png)
    


    
    
    Fri May 13 00:00:00 2011
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 397.6979298591614
    
    in sample:
         RMSE: 6.0469729991068775
         MAE: 4.095968161747656
    
    out of sample:
         RMSE: 8.069629211036991
         MAE: 5.376922319672406
    
    


    
![png](output_10_13.png)
    


    
    
    Wed Oct  5 00:00:00 2011
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 450.1413872241974
    
    in sample:
         RMSE: 5.4729302456629405
         MAE: 3.615352914649668
    
    out of sample:
         RMSE: 7.730246073807314
         MAE: 5.060821509285512
    
    


    
![png](output_10_15.png)
    


    
    
    Wed Feb 29 00:00:00 2012
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 465.2357015609741
    
    in sample:
         RMSE: 5.681673560701141
         MAE: 3.789181475036273
    
    out of sample:
         RMSE: 7.547412776788155
         MAE: 4.903030154281226
    
    


    
![png](output_10_17.png)
    


    
    
    Mon Jul 23 00:00:00 2012
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    cpu: 495.4776117801666
    
    in sample:
         RMSE: 5.465670937794306
         MAE: 3.6999111983069106
    
    out of sample:
         RMSE: 8.33248095414083
         MAE: 5.171770325355775
    
    


    
![png](output_10_19.png)
    


    
    
    Fri Dec 14 00:00:00 2012
    
    training...
    
    alpha: 0.01
    hidden_layer_sizes: (13, 13)
    learning_rate: adaptive
    learning_rate_init: 0.1
    solver: sgd
    early_stopping: False
    max_iter: 500
    warm_start: True
    tol: 0.0001
    
