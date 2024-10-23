import os
import numpy as np
import pandas as pd
from pathlib import Path
prev = Path().resolve().absolute()
print(prev)
script_dir = str(Path(__file__).parent.absolute())
print(f'collecting from: {script_dir}')
os.chdir(script_dir)
df = pd.read_csv([f for f in os.listdir(script_dir) if f.find('bloomberg')!=-1][0]).iloc[:,1:]
frqs = np.sort(df['fixing_frequency'].unique())
nfx = np.sort(df['n_fixings'].unique())
types = np.sort(df['averaging_type'].unique())
T = np.sort(df['days_to_maturity'].unique())
S = np.sort(df['spot_price'].unique())
print('types',types)
print('nfx',nfx)
print('frqs',frqs)
print('T',T)
print('S',S)
os.chdir(prev)
print(os.getcwd())