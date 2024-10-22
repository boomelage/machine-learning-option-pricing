import pandas as pd
import numpy as np
from datetime import datetime
from routine_ivol_collection import raw_calls, raw_puts
from model_settings import ms
from Derman import derman

s = 5625
calculation_datetime = datetime(2024,9,16)
coefficient_calls = raw_calls[raw_calls.index>s]
coefficient_puts = raw_puts[raw_puts.index<s]
coefficient_surface = pd.concat([coefficient_calls,coefficient_puts],ignore_index=False).sort_index(ascending=True)
coefficient_surface = coefficient_surface[coefficient_surface!=0]

T = coefficient_surface.columns
K = coefficient_surface.index
atm_vols = pd.Series(raw_calls.loc[s,T],index=T).dropna()
T = atm_vols.index

derman_coefs = derman(coefficient_surface,s,atm_vols)





print(derman_coefs)