import pandas as pd
import numpy as np
from datetime import datetime
from routine_ivol_collection import raw_calls, raw_puts
from model_settings import ms
from Derman import derman

s = 5625
calculation_datetime = datetime(2024,9,16)
calibration_calls = raw_calls[raw_calls.index>s]
calibration_puts = raw_puts[raw_puts.index<s]
calibration_surface = pd.concat([calibration_calls,calibration_puts],ignore_index=False).sort_index(ascending=True)
calibration_surface = calibration_surface[calibration_surface!=0]

T = calibration_surface.columns
K = calibration_surface.index
atm_vols = pd.Series(raw_calls.loc[s,T],index=T).dropna()
T = atm_vols.index

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
linsurface = derman(calibration_surface,s,atm_vols).replace(0,np.nan).dropna(how='any',axis=1)
linsurface = linsurface[(linsurface>0).all(axis=1)]


T = linsurface.columns
K = linsurface.index


T1 = int(np.max(T[T<=7]))
T2 = int(np.max(T[(T>T1)&(T<=31)]))
T3 = int(np.max(T[(T>T2)&(T<=90)]))
T4 = int(np.max(T[(T>T3)&(T<=180)]))
T5 = int(np.max(T[(T>T4)&(T<=370)]))
T6 = int(np.max(T[(T>T5)&(T<=730)]))
T = np.array([T1,T2,T3,T4,T5,T6],dtype=int)


indices = np.linspace(0, len(linsurface.index) - 1, num=5, dtype=int)
    
K = linsurface.index[indices]

calibration_surface = linsurface.loc[K,T]

print(calibration_surface)


