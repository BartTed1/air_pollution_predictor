import numpy as np
import pandas as pd
import sys
import codecs

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

file_path = "data_3h_Bydgoszcz.csv"

# Wczytanie danych z pliku CSV
data = pd.read_csv(file_path, header=None)

# Nadanie odpowiednich nazw kolumn
data.columns = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]

# Filtracja danych dla miast "Bydgoszcz" i "BydgoszczParkPrzemyslowy" oraz kolumny "pm2_5"
filtered_data = data[data["name"].isin(["Bydgoszcz", "BydgoszczParkPrzemyslowy"])]
filtered_data = filtered_data[["name", "pm2_5"]]

# Wyświetlenie wartości "pm2_5" dla obu miast
pm2_5_values = filtered_data["pm2_5"].tolist()
combined_values = ", ".join(str(value) for value in pm2_5_values)

combined_values_list = combined_values.split(", ")

# Wyświetlenie przekształconych danych
print(combined_values_list)

#====================================================================================
# Przygotowanie danych
pm2_5_numeric = np.array([float(value) for value in combined_values_list])
features = pm2_5_numeric[:-1]
labels = pm2_5_numeric[1:]

# Przygotowanie modelu Random Forests z dostosowanymi parametrami
model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features="sqrt")
model.fit(features.reshape(-1, 1), labels)

# Przewidywanie jednej końcowej wartości
final_value = model.predict(pm2_5_numeric[-1].reshape(1, -1))
print("Finalna przewidziana wartość pm2_5:", final_value[0])

# Przewidywanie wartości dla danych, które nie znajdują się jeszcze w tablicy
predicted_values = model.predict(features.reshape(-1, 1))

# Obliczanie dokładności na podstawie przewidywanych wartości i rzeczywistych wartości
absolute_errors = np.abs(predicted_values - labels)
accuracy = np.mean(absolute_errors)
print("Dokładność (wartość bezwzględna różnicy):", accuracy)