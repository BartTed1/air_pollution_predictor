import sys
import ast

train = 'dane_nauka.csv'
pred = 'dane_predykcja.csv'
results_file = 'new_results.txt'
num_predictions = 10

sys.path.append('./model/3h/RandomForestRegressor')
from forest import RandomForestRegressorModel
forest = RandomForestRegressorModel()

def calculate_error_margin(real_values, predicted_values):
    if len(real_values) != len(predicted_values):
        raise ValueError("Liczba wartości rzeczywistych i przewidywanych musi być taka sama.")
    
    number_values = len(real_values)
    margin_of_error = sum(abs(predicted_values[i] - real_values[i]) / real_values[i] for i in range(number_values))
    percentage_margin_of_error = (margin_of_error / number_values) * 100
    tekst_percentage_margin_of_error = "{:.2f}%".format(percentage_margin_of_error)
    return tekst_percentage_margin_of_error

selected_features = ['pm2_5', 'wind_speed', 'wind_deg', 'temp', 'humidity']

data_train = forest.read_data(train)
data_pred = forest.read_data(pred)

prawdziwe_wartosci_pm2_5 = data_pred['pm2_5']

X_train = forest.select_features(data_train, selected_features)
y_train = data_train['pm2_5_label']
X_pred = forest.select_features(data_pred, selected_features)

with open(results_file, 'r') as file:
    params_str = file.readlines()

params = []
for param_str in params_str:
    params.append(ast.literal_eval(param_str.strip()))

min_average_error_margin = float('inf')
best_line_index = -1

for line_index, param_set in enumerate(params):
    error_margin_sum = 0.0
    
    for _ in range(num_predictions):
        model = forest.train_model(X_train, y_train, param_set)

        predictions = forest.predict(model, X_pred)
        error_margin = calculate_error_margin(prawdziwe_wartosci_pm2_5, predictions)
        error_margin_sum += float(error_margin.strip('%'))

        print(predictions)
        print(error_margin)
        print()
    
    average_error_margin = error_margin_sum / num_predictions
    print("Sredni procentowy margines bledu: {:.2f}%".format(average_error_margin))
    print()
    
    if average_error_margin < min_average_error_margin:
        min_average_error_margin = average_error_margin
        best_line_index = line_index

print("Linia z najmniejszym procentowym marginesem bledu (Linia {}): {:.2f}%".format(best_line_index + 1, min_average_error_margin))
