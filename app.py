from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

# Load the model once when the app starts
model = xgb.Booster()
model.load_model("best_braking_xgboost.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Prepare the data for prediction
        sample = pd.DataFrame({
            'weather': [1 if data['weather'] == 'rainy' else 0],
            'road_type': [1 if data['roadSurface'] == 'asphalt' else 0.7],
            'speed_kmh': [data['speed']],
            'car_mass': [data['mass']],  # Assuming 500kg per unit
            'avg_slope_deg': [data['incline']],
            'max_slope_deg': [data['incline']]
        })
        
        # Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(sample)
        prediction = int(model.predict(dmatrix)[0])
        
        return jsonify({'distance': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)