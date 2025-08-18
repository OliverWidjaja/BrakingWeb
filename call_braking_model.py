import torch
import numpy as np
import pickle
import math

# Import model class definition (copy from notebook)
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BrakingTransformer(torch.nn.Module):
    def __init__(self, input_size, static_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.time_series_embedding = torch.nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.static_embedding = torch.nn.Linear(static_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = torch.nn.Linear(d_model, 1)
        self.fc1 = torch.nn.Linear(d_model * 2, d_model)
        self.fc2 = torch.nn.Linear(d_model, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x_seq, x_static, src_mask=None, src_key_padding_mask=None, return_attn=False):
        x_seq = self.time_series_embedding(x_seq) * math.sqrt(self.d_model)
        x_seq = self.pos_encoder(x_seq)
        transformer_out = self.transformer_encoder(
            x_seq,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        attn_scores = self.attn_pool(transformer_out)
        if src_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(src_key_padding_mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        seq_out = torch.sum(transformer_out * attn_weights, dim=1)
        static_out = self.static_embedding(x_static)
        combined = torch.cat([seq_out, static_out], dim=1)
        out = torch.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return (out, attn_weights.squeeze(-1)) if return_attn else out

def predict_braking_distance(weather, road_type, speed_kmh, car_mass, slope_deg, model_path='best_braking_model.pth', scaler_path='scalers.pkl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrakingTransformer(input_size=4, static_size=4, d_model=64, nhead=4, num_layers=3).to(device)
    # Use weights_only=True for security (PyTorch >=2.3.0)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with open(scaler_path, 'rb') as f:
        # If you see a scikit-learn version warning, try to match the version used for saving and loading.
        scalers = pickle.load(f)
        static_scaler = scalers['static_scaler']
        time_series_scaler = scalers['time_series_scaler']
        max_seq_len = scalers['max_seq_len']
    weather_map = {'ClearNoon': 0, 'WetCloudySunset': 1}
    road_type_map = {'asphalt': 1, 'gravel': 0}
    weather_val = weather_map.get(weather)
    road_type_val = road_type_map.get(road_type)
    if weather_val is None or road_type_val is None:
        raise ValueError('Invalid weather or road_type. Use "ClearNoon" or "WetCloudySunset" for weather, and "asphalt" or "gravel" for road_type.')
    static = np.array([weather_val, road_type_val, speed_kmh, car_mass], dtype=np.float32)
    static = static_scaler.transform([static])[0]
    static = torch.FloatTensor(static).unsqueeze(0).to(device)
    initial_velocity = speed_kmh / 3.6
    avg_deceleration = -7.0
    time_step = 0.0625
    seq_len = min(int(initial_velocity / -avg_deceleration / time_step) + 1, max_seq_len)
    velocity = np.linspace(initial_velocity, 0, seq_len)
    acceleration = np.full(seq_len, avg_deceleration, dtype=np.float32)
    slope = np.full(seq_len, slope_deg, dtype=np.float32)
    time = np.arange(seq_len) * time_step
    time_series = np.column_stack([velocity, acceleration, slope, time])
    if seq_len < max_seq_len:
        padding = np.zeros((max_seq_len - seq_len, 4), dtype=np.float32)
        time_series = np.vstack([time_series, padding])
        mask = np.zeros(max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1
    else:
        mask = np.ones(max_seq_len, dtype=np.float32)
    time_series = time_series_scaler.transform(time_series)
    time_series = torch.FloatTensor(time_series).unsqueeze(0).to(device)
    mask = torch.FloatTensor(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(time_series, static, src_key_padding_mask=(mask == 0))
    return output.item()

if __name__ == "__main__":
    # Example usage
    try:
        braking_distance = predict_braking_distance(
            weather='ClearNoon',
            road_type='asphalt',
            speed_kmh=60,
            car_mass=1903,
            slope_deg=0.03
        )
        print(f'Predicted braking distance: {braking_distance:.2f} m')
    except ValueError as e:
        print(f'Error: {e}')
