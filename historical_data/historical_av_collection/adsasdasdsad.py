# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:44:11 2024

@author: boomelage
"""
from pathlib import Path
from model_settings import ms
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import QuantLib as ql
import time

current_dir = os.path.abspath(str(Path()))

store = pd.HDFStore(r'alphaVantage Vanillas.h5')
keys = store.keys()

