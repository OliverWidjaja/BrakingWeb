
from flask import Flask, request, jsonify
import os
from call_braking_model import predict_braking_distance

app = Flask(__name__)




@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Map incoming data to model's expected input
        # weather: 'ClearNoon' or 'WetCloudySunset'
        # road_type: 'asphalt' or 'gravel'
        weather = data.get('weather', 'ClearNoon')
        road_type = data.get('roadSurface', 'asphalt')
        speed_kmh = data.get('speed', 60)
        car_mass = data.get('mass', 1903)
        slope_deg = data.get('incline', 0.03)
        prediction = predict_braking_distance(
            weather=weather,
            road_type=road_type,
            speed_kmh=speed_kmh,
            car_mass=car_mass,
            slope_deg=slope_deg
        )
        return jsonify({'distance': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)