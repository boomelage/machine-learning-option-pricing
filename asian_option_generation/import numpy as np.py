import numpy as np
import pandas as pd
import QuantLib as ql
from itertools import product


s = 100
K = np.linspace(80,120,50).astype(int).astype(float)


fixing_frequencies = [30,60,90,180,360]
max_maturity = fixing_frequencies[-1]
features_list = []
for f in fixing_frequencies:
	max_periods = np.arange(f,max_maturity+1,f)
	for i, tenor in enumerate(max_periods):
		n_fixings = i+1
		periods = max_periods[:i+1]
		# print(tenor)
		# print(n_fixings)
		# print(f)

		features = pd.DataFrame(
            product(
                [s],
                K,
                [0.04],
                [0.018],
                ['call'],
                ['geometric'],
                [f],
                [n_fixings],
                [0],
                [tenor]
            ),
            columns = [
                'spot_price','strike_price','risk_free_rate','dividend_rate','w',
                'averaging_type','fixing_frequency','n_fixings','past_fixings','days_to_maturity'
            ]
        )

		features_list.append(features)

features = pd.concat(features_list,ignore_index=True)

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(features['n_fixings'].unique())