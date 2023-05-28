import numpy as np
import pandas as pd
import sys
import codecs

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error 

import matplotlib.pyplot as plt

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
# Przygotowanie danych wejściowych
X = np.arange(len(combined_values_list)).reshape(-1, 1)
y = np.array(combined_values_list, dtype=float)

# Inicjalizacja i wytrenowanie modelu SVR
model = SVR()
model.fit(X, y)

# Przewidywanie wartości pm2_5 dla wszystkich poprzednich indeksów
predicted_values = model.predict(X)
predicted_values_formatted = [f"{value:.2f}" for value in predicted_values]

# Wyświetlanie przewidywanych wartości pm2_5 dla wszystkich indeksów
print("Przewidywane wartości pm2_5 dla wszystkich poprzednich indeksów:")
print(predicted_values_formatted)

# Przewidywanie wartości pm2_5 dla kolejnego indeksu
next_index = len(combined_values_list)
next_value = model.predict([[next_index]])

# Obliczanie błędu bezwzględnego średniego (MAE)
mae = mean_absolute_error(y, predicted_values)

# Obliczanie procentowego błędu bezwzględnego średniego (MAE)
mean_values = np.mean(y)
percentage_mae = (mae / mean_values) * 100

print("Przewidywana wartość pm2_5:", next_value[0])
print("Procentowy błąd bezwzględny średni (MAE):", percentage_mae)

#====================================================================================
# OBLICZENIE MARGINESU BLEDU ROZBIEZNOSCI DANYCH

diff = np.abs(np.array(combined_values_list, dtype=float) - np.array(predicted_values_formatted, dtype=float))
mean_error = np.mean(diff)
percent_error = (mean_error / np.mean(np.array(predicted_values_formatted, dtype=float))) * 100
print("Margines błędu między danymi (w procentach):", percent_error)

#====================================================================================
# RYSOWANIE WYKRESU PM2.5

# Konwertowanie wartości na liczby zmiennoprzecinkowe
pm2_5_values = list(map(float, combined_values_list))
predicted_values = list(map(float, predicted_values_formatted))

# Tworzenie wykresu
plt.plot(pm2_5_values, label='PM2.5')
plt.plot(predicted_values, label='Przewidywane dane')
plt.title("Wykres PM2.5 z przewidywanymi danymi")
plt.ylabel("PM2.5")
plt.legend()  # Dodanie legendy
plt.show()
