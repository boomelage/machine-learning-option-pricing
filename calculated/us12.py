import os
import pandas as pd
from pathlib import Path
scr = Path(__file__).resolve().parent

USGG12M = os.path.join(scr,'raw','bloomberg','USGG12M')
f = os.path.join(USGG12M,[f for f in os.listdir(USGG12M) if f.endswith('.csv')][0])

us12 = pd.read_csv(f).iloc[4:,:2].reset_index(drop=True)
us12.columns = us12.iloc[0]
us12 = us12.iloc[1:,:]
us12 = us12.rename(columns={'Dates':'date','LAST_PRICE':'risk_free_rate'})
us12['risk_free_rate'] = pd.to_numeric(us12['risk_free_rate'],errors='coerce')
us12['date'] = pd.to_datetime(us12['date'],format="mixed",errors='coerce')
us12 = us12.set_index('date').dropna().squeeze().sort_index(ascending=False)/100