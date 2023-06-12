import numpy as np
import statsmodels.api as sm
import datetime
import csv


def timestamp_to_day_in_week_number(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.weekday())


def timestamp_to_month_number(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.month)


def timestamp_to_hour(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.hour)


# use data_3h.csv
# Load data from CSV

with open('dane_nauka.csv', 'r') as file:
	csv_reader = csv.reader(file)
	data = []
	labels = []
	for row in csv_reader:
		data.append([
			timestamp_to_day_in_week_number(row[8]),
			timestamp_to_hour(row[8]),
			(1 / float(row[17])) * 10 * (float(row[17]) - float(row[3])),  # temp_change importance
			float(row[6]),  # wind_speed
			float(row[9]),  # pm2_5

		])
		labels.append(float(row[22]))  # pm2_5_label

X_data = np.array(data)
y_data = np.array(labels)

X = sm.add_constant(X_data)

model = sm.OLS(y_data, X)
results = model.fit()
print(results.summary())
