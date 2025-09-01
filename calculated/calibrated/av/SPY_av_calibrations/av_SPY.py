import os
from pathlib import Path
from heston_model_calibration import calibrations
prev = Path().resolve().absolute()
os.chdir(Path(__file__).parent)
calibrations = calibrations.collect_calibrations()
os.chdir(prev)