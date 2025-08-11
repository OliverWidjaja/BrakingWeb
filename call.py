import xgboost as xgb
import pandas as pd

# Load the model from the JSON file
model = xgb.Booster()
model.load_model("best_braking_xgboost.json")

# Prepare the sample data
sample = pd.DataFrame({
    'weather': [1],
    'road_type': [0.7], # 1 or 0.7
    'speed_kmh': [60],
    'car_mass': [1800],
    'avg_slope_deg': [-0.05],
    'max_slope_deg': [0.1]
})

# Convert the sample DataFrame to a DMatrix
dmatrix = xgb.DMatrix(sample)

# Make predictions
predictions = model.predict(dmatrix)

print("Predictions:", predictions)
