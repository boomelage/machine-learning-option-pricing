import os
import pandas as pd
from pathlib import Path
from us12 import us12
scr = Path(__file__).resolve().parent

spx_atm_vols = os.path.join(scr,'raw','bloomberg','SPX')
f = os.path.join(spx_atm_vols,[f for f in os.listdir(spx_atm_vols) if f.endswith('.csv')][0])
spx_ivols = pd.read_csv(f).iloc[3:,:].reset_index(drop=True)
spx_ivols.columns = spx_ivols.iloc[0,:].values
spx_ivols = spx_ivols.iloc[1:]
spx_ivols = spx_ivols.rename(columns={'Dates':'date','PX_LAST':'spot_price','EQY_DVD_YLD_12M':'dividend_rate'})
spx_ivols['date'] = pd.to_datetime(spx_ivols['date'],format='mixed',errors='coerce')
for col in spx_ivols.columns[1:]:
	spx_ivols[col] = pd.to_numeric(spx_ivols[col],errors='coerce')
spx_ivols = spx_ivols.dropna().set_index('date').sort_index(ascending=False)
vol_cols = spx_ivols.columns[2:].tolist()
vol_cols = [c[:c.find('_',0)-2]+'_vol' for c in vol_cols]
spx_ivols.columns = spx_ivols.columns[:2].tolist() + vol_cols
spx_ivols[vol_cols] = spx_ivols[vol_cols]/100
spx_ivols['dividend_rate'] = spx_ivols['dividend_rate']/100
spx_ivols['risk_free_rate'] = us12
spx_ivols = spx_ivols.dropna()