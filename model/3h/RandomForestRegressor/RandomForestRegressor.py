import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint

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

# Inicjalizacja i trening modelu Random Forest Regressor
model = RandomForestRegressor(criterion='absolute_error', 
                              max_depth=15, max_features='sqrt', 
                              min_impurity_decrease=0.0, 
                              min_samples_leaf=4, 
                              min_samples_split=10, 
                              n_estimators=150)
model.fit(X_train, y_train)

# Przewidywanie wartości pm2_5 na podstawie danych do predykcji
predictions = model.predict(X_pred)

# Tworzenie wykresu z wartościami aktualnymi i przewidywanymi
plt.plot(data_pred['pm2_5'], label='Wartości aktualne')
plt.plot(predictions, label='Przewidywane wartości')
plt.xlabel('Indeks próbki')
plt.ylabel('Wartość pm2_5')
plt.title('Porównanie wartości aktualnych i przewidywanych')
plt.legend()

# Przygotowanie danych w formacie [[predykcja, wartosc pm2_5], [predykcja, wartosc pm2_5], ...]
output_data = [[prediction, pm2_5] for prediction, pm2_5 in zip(predictions, data_pred['pm2_5'])]

# Wyświetlenie danych w konsoli
for data in output_data:
    print(data)

# Obliczenie różnicy między wartościami przewidywanymi a wartościami rzeczywistymi
errors = predictions - data_pred['pm2_5']

# Obliczenie procentowego marginesu błędu dla każdej predykcji
percentage_errors = (errors / data_pred['pm2_5']) * 100

# Obliczenie procentowego marginesu błędu łącznego
mean_percentage_error = abs(percentage_errors).mean()

print("Procentowy margines bledu laczny: {:.2f}%".format(mean_percentage_error))

plt.show()

