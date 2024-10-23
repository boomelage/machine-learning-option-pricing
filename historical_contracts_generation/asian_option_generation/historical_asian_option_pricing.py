import time
import pandas as pd
from datetime import datetime
from asian_option_generation import features
from model_settings import asian_option_pricer
aop = asian_option_pricer()

features = features.iloc[:10000].copy()
features = features[features['days_to_maturity']<330]
print(features)

start = time.time()

features['asian']=aop.df_asian_option_price(features)

end = time.time()

print(features)