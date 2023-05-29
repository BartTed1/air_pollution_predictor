import codecs
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class RandomForest:
    def __init__(self):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        self.file_path = "data_3h_Bydgoszcz.csv"

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)

    def set_column_names(self):
        self.data.columns = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                             "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                             "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                             "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]

    def filter_data(self):
        self.filtered_data = self.data[self.data["name"].isin(["Bydgoszcz", "BydgoszczParkPrzemyslowy"])]
        self.filtered_data = self.filtered_data[["name", "pm2_5"]]

    def display_pm2_5_values(self):
        pm2_5_values = self.filtered_data["pm2_5"].tolist()
        combined_values = ", ".join(str(value) for value in pm2_5_values)
        self.combined_values_list = combined_values.split(", ")
        print(self.combined_values_list)

    def prepare_data(self):
        self.pm2_5_numeric = np.array([float(value) for value in self.combined_values_list])
        self.features = self.pm2_5_numeric[:-1]
        self.labels = self.pm2_5_numeric[1:]

    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features="sqrt")
        self.model.fit(self.features.reshape(-1, 1), self.labels)

    def predict_final_value(self):
        self.final_value = self.model.predict(self.pm2_5_numeric[-1].reshape(1, -1))
        print(f"Finalna przewidziana wartość pm2_5: {self.final_value[0]}")

    def predict_values(self):
        self.predicted_values = self.model.predict(self.features.reshape(-1, 1))

    def calculate_accuracy(self):
        self.absolute_errors = np.abs(self.predicted_values - self.labels)
        self.accuracy = np.mean(self.absolute_errors)
        print(f"Dokładność (wartość bezwzględna różnicy): {self.accuracy}")


# RForest = RandomForest()
# RForest.load_data()
# RForest.set_column_names()
# RForest.filter_data()
# RForest.display_pm2_5_values()
# RForest.prepare_data()
# RForest.train_model()
# RForest.predict_final_value
# RForest.predict_values()
# RForest.calculate_accuracy()
