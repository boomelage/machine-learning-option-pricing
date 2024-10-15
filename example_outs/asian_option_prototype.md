```python
import QuantLib as ql
import numpy as np
import pandas as pd
from datetime import datetime
from model_settings import ms
from itertools import product
```

    
    pricing settings:
    Actual/365 (Fixed) day counter
    New York stock exchange calendar
    compounding: continuous
    frequency: annual
    
    


```python
calculation_datetime = datetime.today()
calculation_date = ql.Date(
    calculation_datetime.day,
    calculation_datetime.month,
    calculation_datetime.year
)
r = 0.05
g = 0.02
s = 100.00
k = 95.00
w = 'put'
t = 360

kappa = 0.2
theta = 0.02
rho = -0.75
eta = 0.5
v0 = 0.01 

past_fixings = 0
averaging_frequency = 5
```


```python
my_geometric = ms.ql_asian_price(
            s,k,t,r,g,calculation_datetime, w,
            'geometric',past_fixings,averaging_frequency,
            kappa,theta,rho,eta,v0
            )

my_arithmetic = ms.ql_asian_price(
            s,k,t,r,g,calculation_datetime, w,
            'arithmetic',past_fixings,averaging_frequency,
            kappa,theta,rho,eta,v0
            )
```


```python
vanilla = ms.ql_heston_price(
    s,k,t,r,g,w, 
    kappa, theta, rho, eta, v0, 
    calculation_datetime
)
print(f"\nvanilla: {vanilla}\n\ngeometric: {my_geometric}\n\narithmetic: {my_arithmetic}\n")
```

    
    vanilla: 1.257354183177545
    
    geometric: 0.6976783380092099
    
    arithmetic: 0.6605836170623696
    
    


```python
# K = np.arange(int(s*0.5),int(s)*1.5+1,5).astype(int).tolist()
K = np.arange(s*0.8,s*1.2,5).astype(int).tolist()
averaging_frequencies = [15,180]
T = [180,360]
past_fixings = [0]

features = pd.DataFrame(
    product(
        [s],K,T,
        averaging_frequencies,
        ['call','put'],
        ['geometric','arithmetic'],
        past_fixings,
        [theta],
        [kappa],
        [rho],
        [eta],
        [v0],
        [r],
        [g],
        [calculation_datetime]
    ),
    columns = [ 
        'spot_price',
        'strike_price',
        'days_to_maturity',
        'averaging_frequency',
        'w',
        'averaging_type',
        'past_fixings',
        'theta','kappa','rho','eta','v0',
        'risk_free_rate',
        'dividend_rate',
        'calculation_date',
    ]
)

features['vanilla_price'] = ms.vector_heston_price(features)
features['asian_price'] = ms.vector_asian_price(features)
```


```python
pd.set_option("display.float_format", '{:.6f}'.format)
pd.set_option("display.max_rows",None)
features = features[
    (
        (features['averaging_type']=='arithmetic')
        &
        (features['w']=='put')
    )
]
features[
    [
        'vanilla_price', 'asian_price','spot_price', 'strike_price', 'days_to_maturity', 'averaging_frequency',
        'w', 'averaging_type', 'past_fixings', 'kappa', 'theta', 'rho', 'eta',
        'v0', 'risk_free_rate', 'dividend_rate', 'calculation_date',

    ]
].sort_values(by=['strike_price','days_to_maturity','averaging_frequency','averaging_type','w']).reset_index(drop=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vanilla_price</th>
      <th>asian_price</th>
      <th>spot_price</th>
      <th>strike_price</th>
      <th>days_to_maturity</th>
      <th>averaging_frequency</th>
      <th>w</th>
      <th>averaging_type</th>
      <th>past_fixings</th>
      <th>kappa</th>
      <th>theta</th>
      <th>rho</th>
      <th>eta</th>
      <th>v0</th>
      <th>risk_free_rate</th>
      <th>dividend_rate</th>
      <th>calculation_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.188506</td>
      <td>0.013252</td>
      <td>100.000000</td>
      <td>80</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.188506</td>
      <td>0.183834</td>
      <td>100.000000</td>
      <td>80</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.478264</td>
      <td>0.085212</td>
      <td>100.000000</td>
      <td>80</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.478264</td>
      <td>0.257552</td>
      <td>100.000000</td>
      <td>80</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.327197</td>
      <td>0.049779</td>
      <td>100.000000</td>
      <td>85</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.327197</td>
      <td>0.315814</td>
      <td>100.000000</td>
      <td>85</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.662318</td>
      <td>0.187916</td>
      <td>100.000000</td>
      <td>85</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.662318</td>
      <td>0.392901</td>
      <td>100.000000</td>
      <td>85</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.552411</td>
      <td>0.152293</td>
      <td>100.000000</td>
      <td>90</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.552411</td>
      <td>0.531520</td>
      <td>100.000000</td>
      <td>90</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.910140</td>
      <td>0.353802</td>
      <td>100.000000</td>
      <td>90</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.910140</td>
      <td>0.617904</td>
      <td>100.000000</td>
      <td>90</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.918547</td>
      <td>0.403517</td>
      <td>100.000000</td>
      <td>95</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.918547</td>
      <td>0.869861</td>
      <td>100.000000</td>
      <td>95</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.257354</td>
      <td>0.669782</td>
      <td>100.000000</td>
      <td>95</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.257354</td>
      <td>0.957595</td>
      <td>100.000000</td>
      <td>95</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.582515</td>
      <td>1.130209</td>
      <td>100.000000</td>
      <td>100</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.582515</td>
      <td>1.591627</td>
      <td>100.000000</td>
      <td>100</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.805833</td>
      <td>1.290678</td>
      <td>100.000000</td>
      <td>100</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.805833</td>
      <td>1.555374</td>
      <td>100.000000</td>
      <td>100</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3.856495</td>
      <td>4.160963</td>
      <td>100.000000</td>
      <td>105</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3.856495</td>
      <td>4.023319</td>
      <td>100.000000</td>
      <td>105</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.154582</td>
      <td>3.565619</td>
      <td>100.000000</td>
      <td>105</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3.154582</td>
      <td>3.510641</td>
      <td>100.000000</td>
      <td>105</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8.389481</td>
      <td>8.978609</td>
      <td>100.000000</td>
      <td>110</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8.389481</td>
      <td>8.343495</td>
      <td>100.000000</td>
      <td>110</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6.983793</td>
      <td>8.067453</td>
      <td>100.000000</td>
      <td>110</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6.983793</td>
      <td>7.459343</td>
      <td>100.000000</td>
      <td>110</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>28</th>
      <td>13.203564</td>
      <td>13.860131</td>
      <td>100.000000</td>
      <td>115</td>
      <td>180</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>29</th>
      <td>13.203564</td>
      <td>13.165625</td>
      <td>100.000000</td>
      <td>115</td>
      <td>180</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>30</th>
      <td>11.552855</td>
      <td>12.832951</td>
      <td>100.000000</td>
      <td>115</td>
      <td>360</td>
      <td>15</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
    <tr>
      <th>31</th>
      <td>11.552855</td>
      <td>12.159980</td>
      <td>100.000000</td>
      <td>115</td>
      <td>360</td>
      <td>180</td>
      <td>put</td>
      <td>arithmetic</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.020000</td>
      <td>-0.750000</td>
      <td>0.500000</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>2024-10-15 10:59:23.524446</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.reset_option("display.max_rows")
```
