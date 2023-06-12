import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from matplotlib.widgets import CheckButtons

def read_data(file_path):
    column_names = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                    "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                    "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                    "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]
    data = pd.read_csv(file_path, header=None, names=column_names)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def select_features(data, selected_features):
    return data[selected_features]

def train_model(X_train, y_train, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def predict(model, X_pred):
    return model.predict(X_pred)

def calculate_errors(predictions, actual_values):
    errors = predictions - actual_values
    percentage_errors = (errors / actual_values) * 100
    mean_percentage_error = abs(percentage_errors).mean()
    return mean_percentage_error
