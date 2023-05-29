import numpy as np
import pandas as pd
import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

class SupportVectorRegression:
    def __init__(self):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        self.file_path = "data_3h_Bydgoszcz.csv"

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None)
        self.data.columns = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                             "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                             "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                             "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]

    def filter_data(self):
        filtered_data = self.data[self.data["name"].isin(["Bydgoszcz", "BydgoszczParkPrzemyslowy"])]
        self.filtered_data = filtered_data[["name", "pm2_5"]]

    def display_pm2_5_values(self):
        self.pm2_5_values = self.filtered_data["pm2_5"].tolist()
        self.combined_values_list = ", ".join(str(value) for value in self.pm2_5_values).split(", ")
        print(self.combined_values_list)

    def prepare_input_data(self):
        self.X = np.arange(len(self.combined_values_list)).reshape(-1, 1)
        self.y = np.array(self.combined_values_list, dtype=float)

    def train_model(self):
        self.model = SVR()
        self.model.fit(self.X, self.y)

    def predict_values(self):
        self.predicted_values = self.model.predict(self.X)
        self.predicted_values_formatted = [f"{value:.2f}" for value in self.predicted_values]

    def display_for_every_index(self):
        print(f"Przewidywane wartości pm2_5 dla wszystkich poprzednich indeksów: {self.predicted_values_formatted}")

    def predict_next_value(self):
        self.next_index = len(self.combined_values_list)
        self.next_value = self.model.predict([[self.next_index]])

    def calculate_mae(self):
        self.mae = mean_absolute_error(self.y, self.predicted_values)

    def calculate_percentage_mae(self):
        mean_values = np.mean(self.y)
        self.percentage_mae = (self.mae / mean_values) * 100
        print(f"Przewidywana wartość pm2_5: {self.next_value[0]}")
        print(f"Procentowy błąd bezwzględny średni (MAE): {self.percentage_mae}")

    def calculate_error_margin(self):
        diff = np.abs(np.array(self.combined_values_list, dtype=float) - np.array(self.predicted_values_formatted, dtype=float))
        self.mean_error = np.mean(diff)
        self.percent_error = (self.mean_error / np.mean(np.array(self.predicted_values_formatted, dtype=float))) * 100
        print(f"Margines błędu między danymi (w procentach): {self.percent_error}")

    def chart(self):
        pm2_5_values = list(map(float, self.combined_values_list))
        predicted_values = list(map(float, self.predicted_values_formatted))

        plt.plot(pm2_5_values, label='PM2.5')
        plt.plot(predicted_values, label='Przewidywane dane')
        plt.title("Wykres PM2.5 z przewidywanymi danymi")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.show()


# svr = SupportVectorRegression()
# svr.load_data()
# svr.filter_data()
# svr.display_pm2_5_values()
# svr.prepare_input_data()
# svr.train_model()
# svr.predict_values()
# svr.display_for_every_index()
# svr.predict_next_value()
# svr.calculate_mae()
# svr.calculate_percentage_mae()
# svr.calculate_error_margin()
# svr.chart()
