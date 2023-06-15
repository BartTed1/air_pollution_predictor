import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
sys.path.append('./statistics/3h')
import timeutils as ts

class RandomForestRegressorModel:
    def __init__(self):
        self.data_train = None
        self.data_pred = None
        self.data_train_path = None
        self.data_pred_path = None
        self.model = None
        self.labels = None

    def read_data(self, file_path):
        column_names = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                        "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                        "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                        "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]
        data = pd.read_csv(file_path, header=None, names=column_names)

        return data

    def select_features(self, data, selected_features):
        return data[selected_features]

    def train_model(self, X_train, y_train, params):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_pred):
        return model.predict(X_pred)
    
    def calculate_mean_difference(self, y_true, y_pred):
        difference = np.abs(y_true - y_pred)
        mean_difference = np.mean(difference)
        return mean_difference
