import csv
import utils.time_utils as ts
import neural_network as nn
import test as nnTest

# Load data from CSV
with open('../../data/3h/data_3h_Bydgoszcz3.csv', 'r') as file:
	# Create a CSV reader object
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

# Load evaluate data form CSV
with open('../../data/3h/data_3h_Bydgoszcz_test3.csv', 'r') as file:
	# Create a CSV reader object
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

test = nnTest.NNTest(neural_network, data2, labels2)
print(test.forecast(5))

# Save the neural network
neural_network.save()

# Load the neural network
neural_network.load('model.h5')

# Use the neural network to make predictions
# neural_network.predict()
