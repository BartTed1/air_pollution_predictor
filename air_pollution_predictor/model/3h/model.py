import csv
import utils.help_times as ts
from neural_network import NeuralNetwork as nn

# Load data from CSV
with open('../../data/3h/data_3h_Bydgoszcz.csv', 'r') as file:
	# Create a CSV reader object
	csv_reader = csv.reader(file)
	data = []
	labels = []
	# Print the array
	for row in csv_reader:
		data.append([
			ts.timestamp_to_day_in_week_number(row[8]),
			ts.timestamp_to_hour(row[8]),
			(1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
			float(row[6]),  # wind_speed
			float(row[9]),  # pm2_5
		])
		labels.append(float(row[22]))  # pm2_5_label

# Create a neural network object
neural_network = nn.NeuralNetwork(data, labels)

# Load the neural network
neural_network.load('model.h5')

# Build and train the neural network
neural_network.build_and_train(data, labels)

# Use the neural network to make predictions
neural_network.predict()

# Save the neural network
neural_network.save()