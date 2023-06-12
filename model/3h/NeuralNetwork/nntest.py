import neural_network as NeuralNetwork
from statistics import mean

class NNTest:
	"""
	Tests the neural network
	:param nn: NeuralNetwork object
	:param data: []
	:param labels: []
	"""

	def __init__(self, nn: NeuralNetwork, data: [], labels: []):
		self.model: NeuralNetwork = nn
		self.data = data
		self.labels = labels
		self.correct = 0
		self.incorrect = 0

	def variance(self) -> []:
		variance = []
		for i in range(len(self.data)):
			predicted = self.model.predict([self.data[i]])
			actual = self.data[i][4]
			future = self.labels[i]
			real_change = future - actual
			predicted_change = predicted - actual
			variance.append(abs(predicted_change - real_change))
		return variance

	def mean_variance(self) -> float:
		variance = self.variance()
		return sum(variance) / len(variance)

	def max_variance(self) -> float:
		variance = self.variance()
		return max(variance)

	def min_variance(self) -> float:
		variance = self.variance()
		return min(variance)

	def variance_accuracy(self) -> float:
		accuracy = []
		for i in range(len(self.data)):
			predicted = self.model.predict([self.data[i]])
			actual = self.data[i][4]
			future = self.labels[i]
			real_change = future - actual
			if real_change == 0:
				continue
			predicted_change = predicted - actual
			accuracy.append(abs(predicted_change - real_change) / real_change)
		return mean(accuracy)

	def trend(self) -> []:
		"""
		Compares the predicted trend with the real trend
		:return: [correct, incorrect] number of correct and incorrect predictions
		"""
		trend = [0, 0]  # [correct, incorrect]
		for i in range(len(self.data)):
			predicted = self.model.predict([self.data[i]])
			actual = self.data[i][4]
			future = self.labels[i]
			real_change = (future - actual) > 0
			predicted_change = (predicted - actual) > 0
			if predicted_change == real_change:
				trend[0] += 1
			else:
				trend[1] += 1
		return trend

	def trend_accuracy(self):
		trend = self.trend()
		return trend[0] / (trend[0] + trend[1])
	

	def forecast(self, times: int) -> [[], []]:
			"""
			Creating a forecast for the given times
			Need continuous time data
			:param times: number of times to forecast
			:return: [predicted, actual] predicted and actual values
			"""
			predicted = [self.data[0][4]]
			actual = []
			result = []
			for i in range(times):
				predicted.append(self.model.predict([self.data[i][:-1] + [predicted[-1]]]))
				actual.append(self.labels[i])
				result.append([predicted[-1], actual[-1]])
				# trend
				if i < 1:
					continue
				predicted_change = (predicted[-1] - predicted[-2]) > 0
				actual_change = (actual[-1] - actual[-2]) > 0
				if predicted_change == actual_change:
					self.correct += 1
				else:
					self.incorrect += 1
			return result

	def pobierz_trendy(self):
		return self.correct, self.incorrect	
