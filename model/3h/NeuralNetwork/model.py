import csv
import timeutils as ts
import neural_network as nn
import test as nnTest
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
with open('dane_nauka.csv', 'r') as file:
    csv_reader = csv.reader(file)
    data = []
    labels = []
    for row in csv_reader:
        data.append([
            ts.timestamp_to_day_in_week_number(row[8]),
            ts.timestamp_to_hour(row[8]),
            (1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
            float(row[6]),  # wind_speed
            float(row[9]),  # pm2_5
        ])
        labels.append(float(row[22]))  # pm2_5_label

# Load evaluate data from CSV
with open('dane_predykcja.csv', 'r') as file:
    csv_reader = csv.reader(file)
    data2 = []
    labels2 = []
    for row in csv_reader:
        data2.append([
            ts.timestamp_to_day_in_week_number(row[8]),
            ts.timestamp_to_hour(row[8]),
            (1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
            float(row[6]),  # wind_speed
            float(row[9]),  # pm2_5
        ])
        labels2.append(float(row[22]))  # pm2_5_label

# Create a neural network object
neural_network = nn.NeuralNetwork(data, labels)

# Build and train the neural network
neural_network.build_and_train()

# Create a test object
test = nnTest.NNTest(neural_network, data2, labels2)

#print(data)

# Print predictions
#print(predictions)

# Save the neural network
neural_network.save()

# Load the neural network
neural_network.load('model.h5')

# Use the neural network to make predictions
# neural_network.predict()

# Print predictions
predictions = test.forecast(12)
print(predictions)

# Convert predictions to a list of pm2.5 values
pm25_predictions = [pred[0] for pred in predictions]

# Plot the data
x = range(len(labels2))
plt.plot(x, labels2, label='Actual Values')
plt.plot(x, pm25_predictions, label='Predictions')
plt.xlabel('Sample')
plt.ylabel('pm2.5')
plt.title('Actual Values vs Predictions')
plt.legend()
plt.show()
