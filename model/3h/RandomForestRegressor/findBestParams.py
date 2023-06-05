import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Wczytanie danych z pliku CSV do nauki modelu
column_names = ["name", "lat", "lon", "temp", "humidity", "pressure", "wind_speed", "wind_deg", "timestamp",
                "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3", "temp_label", "humidity_label",
                "pressure_label", "wind_speed_label", "wind_deg_label", "pm2_5_label", "pm10_label", "co_label",
                "no_label", "no2_label", "o3_label", "so2_label", "nh3_label"]
data_train = pd.read_csv('dane_nauka.csv', header=None, names=column_names)

# Dodanie kolumny z timestampem
data_train['timestamp'] = pd.to_datetime(data_train['timestamp'])

# Wczytanie danych z pliku CSV do predykcji
data_pred = pd.read_csv('dane_predykcja.csv', header=None, names=column_names)

# Dodanie kolumny z timestampem
data_pred['timestamp'] = pd.to_datetime(data_pred['timestamp'])

# Wybór wybranych cech
selected_features = ['pm2_5', 'wind_speed', 'wind_deg', 'temp', 'humidity']

# Obliczenie różnicy czasu między kolejnymi wierszami w danych do nauki
data_train['time_diff'] = (data_train['timestamp'] - data_train['timestamp'].shift()).fillna(pd.Timedelta(hours=3))

# Obliczenie różnicy czasu między kolejnymi wierszami w danych do predykcji
data_pred['time_diff'] = (data_pred['timestamp'] - data_pred['timestamp'].shift()).fillna(pd.Timedelta(hours=3))

# Podział danych na cechy (X) i wartości oczekiwane (y) dla danych do nauki modelu
X_train = data_train[selected_features]
y_train = data_train['pm2_5_label']

# Podział danych na cechy (X) dla danych do predykcji
X_pred = data_pred[selected_features]

#!========================================================================
# DOPASOWANIE ODPOWIEDNICH PARAMETROW
#!========================================================================

param_grid = {
    'n_estimators' : [50, 100, 150, 200],
    'criterion' : ['squared_error', 'absolute_error'],
    'max_depth' : [None, 5, 10, 15],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features' : ['sqrt', 'log2'],
    'min_impurity_decrease' : [0.0, 0.1, 0.2]
}

# Inicjalizacja modelu
model = RandomForestRegressor()

# Liczba powtorzen petli
num_iterations = 5

# Inicjalizacja pustej listy na wyniki
results = []

for i in range(num_iterations):
    # Inicjalizacja GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)

    # Dopasowanie modelu do danych treningowych
    grid_search.fit(X_train, y_train)

    # Wybor najlepszych parametrow
    best_params = grid_search.best_params_

    # Dodanie wynikow do listy
    result = f"{i+1}: {best_params}"
    results.append(result)

    # Wypisanie wynikow w konsoli
    print(result)

with open('results.txt', 'w') as file:
    for result in results:
        file.write(result + "\n")
