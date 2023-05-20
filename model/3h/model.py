import json
import tensorflow as tf
import numpy as np
import datetime
import csv


def unix_timestamp_to_day_seconds(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int((date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())


def timestamp_to_day_in_week_number(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.weekday())


def timestamp_to_hour(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.hour)


# Load data from CSV
with open('../../data/3h/data_3h_Bydgoszcz.csv', 'r') as file:
	# Create a CSV reader object
	csv_reader = csv.reader(file)
	data = []
	labels = []
	# Print the array
	for row in csv_reader:
		data.append([
			timestamp_to_day_in_week_number(row[8]),
			timestamp_to_hour(row[8]),
			(1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
			float(row[6]),  # wind_speed
			float(row[9]),  # pm2_5
		])
		labels.append(float(row[22]))  # pm2_5_label


# Define the input and output dimensions
input_dim = 5  # number of features
output_dim = 1  # number of output labels

# Define the layers of the neural network
model = tf.keras.Sequential([
	tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(32, activation='linear'),
	tf.keras.layers.Dense(output_dim)
])

# Compile the model with an appropriate loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer)

# Train the model on the provided data-label set
model.fit(data, labels, epochs=100, verbose=1)

# Use the trained model to make predictions on new data
new_data = [[3, 19, 0.2434, 5.41, 5.71]]
predictions = model.predict(new_data)
print(predictions)