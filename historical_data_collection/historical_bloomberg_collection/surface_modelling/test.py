from heston_model_calibration import calibrate_heston
from historical_bloomberg_surface_modelling import calibration_surface, s

params = calibrate_heston(calibration_surface,s,0.04,0.018)

print(params)