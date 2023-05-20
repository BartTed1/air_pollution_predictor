import datetime

def timestamp_to_day_in_week_number(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.weekday())

def timestamp_to_month_number(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.month)

def timestamp_to_hour(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int(date.hour)

def unix_timestamp_to_day_seconds(timestamp) -> int:
	date = datetime.datetime.fromtimestamp(int(timestamp))
	return int((date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())