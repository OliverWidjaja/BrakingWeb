from call_braking_model import predict_braking_distance

if __name__ == "__main__":
    # Example input matching the frontend's expected values
    weather = "ClearNoon"  # or "WetCloudySunset"
    road_type = "gravel"  # or "gravel"
    speed_kmh = 50
    car_mass = 2000
    slope_deg = 0.03

    try:
        distance = predict_braking_distance(
            weather=weather,
            road_type=road_type,
            speed_kmh=speed_kmh,
            car_mass=car_mass,
            slope_deg=slope_deg
        )
        print(f"Predicted braking distance: {distance:.2f} m")
    except Exception as e:
        print(f"Error: {e}")
