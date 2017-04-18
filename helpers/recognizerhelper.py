from sklearn import preprocessing

from models.opencv_method_distance import OpencvMethodDistance
import numpy as np


class RecognizeHelper(object):
	@staticmethod
	def calculate_percentage_for_opencv_methods(method, distance, reserve=False):
		percentage = 0

		method_model = OpencvMethodDistance.query.filter(OpencvMethodDistance.code == method).first()

		if method_model is not None:
			per100 = abs(method_model.best - method_model.worst)

			if method_model.worst < 0:
				distance = distance + abs(method_model.worst)

			percentage = (distance / per100) * 100

			print("(", distance, "/", per100, ") * 100")
			print(percentage)

			if not reserve:
				percentage = 100 - percentage

		return percentage

	@staticmethod
	def calculate_percentage_for_distance_metric_methods(user_id, recognized_distance,  distances):

		only_dist = []
		for i, dist in enumerate(distances):
			only_dist.append(dist[0])

		percentage = (float("{0:.5f}".format((1 - (recognized_distance / max(only_dist))) * 100)))

		return percentage


	@staticmethod
	def normalize_data(X_train, X_test):
		print("Normalize data sets")
		std_scale = preprocessing.Normalizer(norm="max").fit(X_train)
		X_pca = std_scale.transform(X_train)
		X_test = std_scale.transform(X_test)

		return X_pca, X_test


