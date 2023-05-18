import tensorflow as tf
import datetime
import csv


def unix_timestamp_to_day_seconds(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int((date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())


# Load data from CSV
with open('../../data/3h/data_3h.csv', 'r') as file:
	# Create a CSV reader object
	csv_reader = csv.reader(file)
	data = []
	labels = []
	# Print the array
	for row in csv_reader:
		data.append([
			float(row[1]),  # lat
			float(row[2]),  # lon
			unix_timestamp_to_day_seconds(int(row[8])),  # seconds since midnight
			float(row[3]),  # temp
			float(row[17]),  # temp_label
			float(row[4]),  # humidity
			float(row[18]),  # humidity_label
			float(row[6]),  # wind_speed
			float(row[20]),  # wind_speed_label
			float(row[9]),  # pm2_5
		])
		labels.append(float(row[22]))  # pm2_5_label


# Define the input and output dimensions
input_dim = 10  # number of features
output_dim = 1  # number of output labels

# Define the layers of the neural network
model = tf.keras.Sequential([
	tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(32, activation='linear'),
	tf.keras.layers.Dense(output_dim)
])

# Compile the model with an appropriate loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(loss='mse', optimizer=optimizer)

# Train the model on the provided data-label set
model.fit(data, labels, epochs=100, verbose=1)

# Use the trained model to make predictions on new data
new_data = [[53.1342, 17.9943, 68963, 286.26, 283.93, 49, 59, 7.2, 4.12, 0.97]]
predictions = model.predict(new_data)

print(predictions)