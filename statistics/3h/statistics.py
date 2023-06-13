import numpy as np
import statsmodels.api as sm
import csv
import timeutils

# use data_3h.csv
# Load data from CSV
with open('../../data/3h/data_3h_Bydgoszcz.csv', 'r') as file:
	csv_reader = csv.reader(file)
	data = []
	labels = []
	for row in csv_reader:
		data.append([
			timeutils.timestamp_to_day_in_week_number(row[8]),
			timeutils.timestamp_to_hour(row[8]),
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
