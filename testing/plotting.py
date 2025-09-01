from model_settings import ms
from pathlib import Path
import pandas as pd
import os
from time import time
tic = time()
ms.find_root(Path())
models_dir = os.path.join(ms.root,ms.trained_models)
models = pd.Series([f for f in os.listdir(models_dir) if not f.startswith('.') and f.find('Legacy')])
for i,m in enumerate(models):
    print(f"{i}     {m}")

# i = input("select model index: ")
selected_model = models[int(3)]

directory = os.path.join(models_dir,selected_model)

import joblib 
model = joblib.load(directory)