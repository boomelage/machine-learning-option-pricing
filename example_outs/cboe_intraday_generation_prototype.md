```python
import os
import numpy as np
import pandas as pd
import QuantLib as ql
from pathlib import Path
from model_settings import ms
from heston_model_calibration import calibrate_heston
datafile = [f for f in os.listdir(Path().resolve()) if f.endswith('.csv')][0]
raw = pd.read_csv(datafile)
raw = raw[
    [
        'underlying_symbol', 'quote_datetime', 'sequence_number', 'root',
        'expiration', 'strike', 'option_type', 'trade_size',
        'trade_price',
        'best_bid', 'best_ask', 'trade_iv', 'trade_delta', 'underlying_bid',
    ]
]
raw = raw.rename(columns={'strike':'strike_price','option_type':'w'})
raw.dtypes
```




    underlying_symbol     object
    quote_datetime        object
    sequence_number        int64
    root                  object
    expiration            object
    strike_price         float64
    w                     object
    trade_size             int64
    trade_price          float64
    best_bid             float64
    best_ask             float64
    trade_iv             float64
    trade_delta          float64
    underlying_bid       float64
    dtype: object




```python
df = raw.copy()
df['quote_datetime'] = pd.to_datetime(df['quote_datetime']).dt.round('h')
df['expiration'] = pd.to_datetime(df['expiration'],format='%Y-%m-%d')
df['days_to_maturity'] = (df['expiration'] - df['quote_datetime']) / pd.Timedelta(days=1)
df['days_to_maturity'] = df['days_to_maturity'].astype(int)
df = df[df['days_to_maturity']>0]
df[['strike_price','underlying_bid']] = df[['strike_price','underlying_bid']].replace(0,np.nan)
df['w'] = df['w'].replace({'C': 'call', 'P': 'put'})
df = df.dropna().reset_index(drop=True)
df['moneyness'] = ms.vmoneyness(df['underlying_bid'],df['strike_price'],df['w'])
df = df[(df['moneyness']<0)&(df['moneyness']>=-0.5)]
df.dtypes
```




    underlying_symbol            object
    quote_datetime       datetime64[ns]
    sequence_number               int64
    root                         object
    expiration           datetime64[ns]
    strike_price                float64
    w                            object
    trade_size                    int64
    trade_price                 float64
    best_bid                    float64
    best_ask                    float64
    trade_iv                    float64
    trade_delta                 float64
    underlying_bid              float64
    days_to_maturity              int64
    moneyness                   float64
    dtype: object




```python
trading_hours = df['quote_datetime'].drop_duplicates().sort_values()
trading_hours = trading_hours[trading_hours.dt.hour >= 9].reset_index(drop=True)
trading_hours
df = df[df['quote_datetime'].isin(trading_hours)].copy().reset_index(drop=True)
for hi,h in enumerate(trading_hours):
    subset = df[df['quote_datetime']==trading_hours[hi]].copy()
    subset = subset.sort_values(by='trade_size',ascending=False)
    subset = subset[subset['trade_size']>= subset['trade_size'].describe()['75%']].reset_index(drop=True)
    s = np.mean(subset['underlying_bid'])
    K = np.sort(subset['strike_price'].unique())
    K = K[np.linspace(0,len(K)-1,5,dtype=int)]
    T = np.sort(subset['days_to_maturity'].unique())
    T = T[T>=5]
    T = T[np.linspace(0,len(T)-1,5,dtype=int)]
    T
    pivoted = subset.pivot_table(index='strike_price',columns='days_to_maturity',values=['trade_iv'],aggfunc='mean')
    r = 0.04
    g = 0.0
    calibration_contracts = pivoted.unstack().reset_index().dropna().reset_index(drop=True).iloc[:,1:]
    calibration_contracts = calibration_contracts.pivot(columns='days_to_maturity',index='strike_price').reset_index().set_index('strike_price')
    T = [i[1] for i in calibration_contracts.columns.tolist()]
    calibration_contracts.columns = T
    heston_parameters = pd.Series(calibrate_heston(calibration_contracts,s,r,g))
    print(h)
    print(heston_parameters)
```

    2024-09-13 09:00:00
    theta     0.024505
    kappa    49.964886
    eta       4.396851
    rho      -0.469353
    v0        0.011397
    dtype: float64
    2024-09-13 10:00:00
    theta    0.049858
    kappa    6.872323
    eta      2.273607
    rho     -0.624556
    v0       0.014515
    dtype: float64
    2024-09-13 11:00:00
    theta    0.048536
    kappa    7.773544
    eta      2.285714
    rho     -0.590700
    v0       0.010746
    dtype: float64
    2024-09-13 12:00:00
    theta     0.047554
    kappa    10.949069
    eta       2.289693
    rho      -0.562720
    v0        0.008310
    dtype: float64
    2024-09-13 13:00:00
    theta     0.040735
    kappa    11.489219
    eta       2.181049
    rho      -0.575582
    v0        0.008367
    dtype: float64
    2024-09-13 14:00:00
    theta     0.037789
    kappa    13.332683
    eta       2.291023
    rho      -0.569487
    v0        0.007702
    dtype: float64
    2024-09-13 15:00:00
    theta    3.805110e-02
    kappa    3.768182e+01
    eta      4.456942e+00
    rho     -5.630132e-01
    v0       2.117336e-10
    dtype: float64
    2024-09-13 16:00:00
    theta    3.585067e-02
    kappa    3.752277e+01
    eta      4.167832e+00
    rho     -5.416538e-01
    v0       1.017338e-09
    dtype: float64
    2024-09-13 17:00:00
    theta    4.717044e-02
    kappa    4.945717e+01
    eta      2.653243e+00
    rho     -5.434059e-01
    v0       9.597664e-10
    dtype: float64
    
