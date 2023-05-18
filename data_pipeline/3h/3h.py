import argparse
import datetime
import sqlite3
import requests
import threading
from time import sleep


def fetch(api_key, weather_response, pollution_response, lat, lon, name):
	"""
	Recurrence function that fetches data from the API and writes it to the database
	:param api_key: OpenWeatherMap API key
	:param weather_response: response from the weather API
	:param pollution_response: response from the pollution API
	:param lat: latitude of the location
	:param lon: longitude of the location
	"""
	weather = weather_response.json()
	pollution = pollution_response.json()

	# sleep 3 hours
	sleep(10800)
	try:
		weather_labels_response = requests.get(
			f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}")
		pollution_labels_response = requests.get(
			f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}")
	except Exception as e:
		print("Nie udało się pobrać danych")
		return fetch(api_key, weather_response, pollution_response, lat, lon, name)
	print("Pobrano dane (fetch)")
	# Convert responses to JSON and create a dataset
	forecast = weather_labels_response.json()
	pollution_labels = pollution_labels_response.json()

	dataset = {
		"name": name,
		"lat": lat,
		"lon": lon,
		"temp": weather["main"]["temp"],
		"humidity": weather["main"]["humidity"],
		"pressure": weather["main"]["pressure"],
		"wind_speed": weather["wind"]["speed"],
		"wind_deg": weather["wind"]["deg"],
		"timestamp": weather["dt"],
		"pm2_5": pollution["list"][0]["components"]["pm2_5"],
		"pm10": pollution["list"][0]["components"]["pm10"],
		"co": pollution["list"][0]["components"]["co"],
		"no": pollution["list"][0]["components"]["no"],
		"no2": pollution["list"][0]["components"]["no2"],
		"o3": pollution["list"][0]["components"]["o3"],
		"so2": pollution["list"][0]["components"]["so2"],
		"nh3": pollution["list"][0]["components"]["nh3"],
		"temp_label": forecast["main"]["temp"],
		"humidity_label": forecast["main"]["humidity"],
		"pressure_label": forecast["main"]["pressure"],
		"wind_speed_label": forecast["wind"]["speed"],
		"wind_deg_label": forecast["wind"]["deg"],
		"pm2_5_label": pollution_labels["list"][0]["components"]["pm2_5"],
		"pm10_label": pollution_labels["list"][0]["components"]["pm10"],
		"co_label": pollution_labels["list"][0]["components"]["co"],
		"no_label": pollution_labels["list"][0]["components"]["no"],
		"no2_label": pollution_labels["list"][0]["components"]["no2"],
		"o3_label": pollution_labels["list"][0]["components"]["o3"],
		"so2_label": pollution_labels["list"][0]["components"]["so2"],
		"nh3_label": pollution_labels["list"][0]["components"]["nh3"],
	}

	requests.post("LINK TO DATABASE CONTROLLER", data=dataset)
	print("Zapisano dane")

	fetch(api_key, weather_labels_response, pollution_labels_response, lat, lon,name)


def main(name, lat, lon):
	"""
	Function that initializes the process of downloading data from the OpenWeatherMap API.
	:param name: name of the process (e.g. location, city...)
	:param lat: latitude of the location
	:param lon: longitude of the location
	"""
	with open("./data_pipeline/api_key.txt", "r") as f:
		api_key = f.read()
		if api_key == "":
			raise Exception("OpenWeatherMap API key is empty")
	try:
		weather_response = requests.get(
			f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}")
		pollution_response = requests.get(
			f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}")
	except Exception as e:
		print("Nie udało się pobrać danych")
		return main(name, lat, lon)
	print("Pobrano dane (main)")
	fetch(api_key, weather_response, pollution_response, lat, lon, name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fetch data from OpenWeatherMap API')
	parser.add_argument('-name', type=str, help='Name of the process (e.g. location, city...)', required=True)
	parser.add_argument('-lon', type=float, help='Longitude of the location', required=True)
	parser.add_argument('-lat', type=float, help='Latitude of the location', required=True)
	args = parser.parse_args()

	thread1 = threading.Thread(target=main, args=(args.name, args.lat, args.lon))
	thread2 = threading.Thread(target=main, args=(args.name, args.lat, args.lon))
	thread3 = threading.Thread(target=main, args=(args.name, args.lat, args.lon))

	print("thread1")
	thread1.start()
	sleep(3600)
	print("thread2")
	thread2.start()
	sleep(3600)
	print("thread3")
	thread3.start()

	thread1.join()
	thread2.join()
	thread3.join()

