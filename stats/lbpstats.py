from collections import Counter
from operator import itemgetter

import cv2
import numpy
from decimal import Decimal

import time
from flask import json, g
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, SVC

from helpers.imagehelper import ImageHelper
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser, ResponseParser
from models.histogram import Histogram
from models.image import Image


class LBPStats:
	OPENCV_METHODS = {
		"correlation": cv2.HISTCMP_CORREL,
		"chi-squared": cv2.HISTCMP_CHISQR,
		"intersection": cv2.HISTCMP_INTERSECT,
		"bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
		"none": None
	}

	SCIPY_METHODS = {
		"euclidean": dist.euclidean,
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, num_points=24, radius=8, method='uniform', algorithm='chi-squared'):
		self.points = int(num_points)
		self.radius = int(radius)
		self.method = str(method)
		self.input_parser = InputParser()
		self.algorithm = algorithm

	def check_svm(self):
		true_positive_rate = []
		false_positive_rate = []
		TPR_d = []
		FPR_d = []

		TPR_m = []
		FPR_m = []

		roc_auc_max = 0

		distances = []
		score = []
		y = []

		for x in range(0, 1):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			param_grid = {
				'C': [1, 5, 10, 50, 100],
				'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
			}

			clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, n_jobs=1, scoring='roc_auc')
			clf = clf.fit(train_data, train_labels)
			s = clf.score(test_data, test_labels)

			# for i, hist_test in enumerate(test_data):
			# 	print(hist_test)
			# 	print(hist_test.reshape(1, -1))
			# 	# p = clf.predict(hist_test.reshape(1, -1))
			# 	t = []
			#
			# 	t2 = []
			# 	t2.append(test_labels[i])
			# 	# print(t)
			# 	s = clf.score(hist_test.reshape(1, -1), t2)
			# 	# print(p)
			# 	print(s)
			# 	score.append(s)
		exit()
		min_score = min(score)
		max_score = max(score)
		threshold = numpy.linspace(min_score, max_score, 50)
		roc_x_matrix = []
		roc_y_matrix = []

		for x in range(0, 5):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			roc_x = []
			roc_y = []

			param_grid = {
				'C': [1, 5, 10, 50, 100],
				'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
			}

			clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, n_jobs=10)
			clf = clf.fit(train_data, train_labels)

			for (thr_index, thr) in enumerate(threshold):

				FP = 0
				TP = 0
				FN = 0
				TN = 0

				for i, hist_test in enumerate(test_data):
					distance = clf.score(hist_test)
					found_ID = clf.predict(hist_test)

					if int(found_ID) == int(test_labels[i]):
						if distance > thr:
							FN = FN + 1
						else:
							TP = TP + 1
					else:
						if distance < thr:
							FP = FP + 1
						else:
							TN = TN + 1

				try:
					TPR = TP / (TP + FN)
				except ZeroDivisionError:
					TPR = 0
				try:
					FPR = FP / (FP + TN)
				except ZeroDivisionError:
					FPR = 0

				roc_x.append(FPR)
				roc_y.append(TPR)

			roc_auc = auc(roc_x, roc_y)

			print(x, " - ROC AUC: ", roc_auc)

			if roc_auc > 0.40:
				roc_x_matrix.append(roc_x)
				roc_y_matrix.append(roc_y)

		newTPR = []
		newFPR = []
		for i in range(0, 50):
			values = []
			valuesFalse = []
			for vector in roc_y_matrix:
				values.append(vector[i])
			for vector in roc_x_matrix:
				valuesFalse.append(vector[i])

			newTPR.append(numpy.mean(values))
			newFPR.append(numpy.mean(valuesFalse))

		roc_auc = auc(newFPR, newTPR)
		print("AUC NEW:", roc_auc)

		plt.figure()
		label = "AUC = %0.2f\n" % roc_auc
		label += "Radius = %i " % self.radius
		label += "Points = %i " % self.points
		label += "Method = %s " % self.method
		plt.plot(newFPR, newTPR, label=label)
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		filename = 'statistics/lbp-roc-a-' + self.algorithm + '-p-' + str(self.points) + '-r-'+ str(self.radius) + '-m-' + self.method +'.png'
		# plt.savefig(filename)
		ImageHelper.save_plot_image(plt, 'roc', g.user.id)

		newTPR = []
		newFPR = []


	def check_distances(self):
		reverse = False
		# if we are using the correlation or intersection
		# method, then sort the results in reverse order
		if self.algorithm in ("correlation", "intersection", "bhattacharyya"):
			reverse = True

		score = []
		y = []
		for x in range(0, 10):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			for i, hist_test in enumerate(test_data):
				for j, hist_train in enumerate(train_data):
					distance = self.calculate_distance(hist_test, hist_train)
					score.append(distance)
					y.append(train_labels[j])

		min_score = min(score)
		max_score = max(score)
		threshold = numpy.linspace(min_score, max_score, 50)
		roc_x_matrix = []
		roc_y_matrix = []

		for x in range(0, 5):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			roc_x = []
			roc_y = []

			for (thr_index, thr) in enumerate(threshold):

				FP = 0
				TP = 0
				FN = 0
				TN = 0

				for i, hist_test in enumerate(test_data):
					distances = []
					for j, hist_train in enumerate(train_data):
						distance = self.calculate_distance(hist_test, hist_train)
						distances.append((distance, train_labels[j]))

					if reverse:
						found_ID = max(distances)[1]
						distance = max(distances)[0]
					else:
						found_ID = min(distances)[1]
						distance = min(distances)[0]

					if int(found_ID) == int(test_labels[i]):
						if reverse:
							if distance < thr:
								FN = FN + 1
							else:
								TP = TP + 1
						else:
							if distance > thr:
								FN = FN + 1
							else:
								TP = TP + 1
					else:
						if reverse:
							if distance > thr:
								FP = FP + 1
							else:
								TN = TN + 1
						else:
							if distance < thr:
								FP = FP + 1
							else:
								TN = TN + 1

				try:
					TPR = TP / (TP + FN)
				except ZeroDivisionError:
					TPR = 0
				try:
					FPR = FP / (FP + TN)
				except ZeroDivisionError:
					FPR = 0

				roc_x.append(FPR)
				roc_y.append(TPR)

			roc_auc = auc(roc_x, roc_y)

			print(x, " - ROC AUC: ", roc_auc)

			if roc_auc > 0.40:
				roc_x_matrix.append(roc_x)
				roc_y_matrix.append(roc_y)

		newTPR = []
		newFPR = []
		for i in range(0, 50):
			values = []
			valuesFalse = []
			for vector in roc_y_matrix:
				values.append(vector[i])
			for vector in roc_x_matrix:
				valuesFalse.append(vector[i])

			newTPR.append(numpy.mean(values))
			newFPR.append(numpy.mean(valuesFalse))

		roc_auc = auc(newFPR, newTPR)
		print("AUC NEW:", roc_auc)
		filename = 'LBP ROC curve'
		plt.figure()
		plt.title(filename)
		label = "AUC = %0.2f\n" % roc_auc
		label += "Radius = %i \n" % self.radius
		label += "Points = %i \n" % self.points
		label += "Method = %s \n" % self.method
		label += "Algorithm = %s " % self.algorithm
		plt.plot(newFPR, newTPR, label=label)
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		image_id = ImageHelper.save_plot_image(plt, 'roc', g.user.id)
		ResponseParser().add_image('roc', 'roc_graph', image_id)

	def calculate_distance(self, test, train):

		train = HistogramMaker.np_hist_to_cv(train)
		test = HistogramMaker.np_hist_to_cv(test)

		switcher = {
			# "svm": self.svm_recognize,
			"correlation": self.opencv_recognize_method,
			"chi-squared": self.opencv_recognize_method,
			"intersection": self.opencv_recognize_method,
			"bhattacharyya": self.opencv_recognize_method,
			"euclidean": self.scipy_recognize_method,
			"manhattan": self.scipy_recognize_method,
			"chebysev": self.scipy_recognize_method,
			"cosine": self.scipy_recognize_method,
			"braycurtis": self.scipy_recognize_method,
		}

		# Get the function from switcher dictionary
		func = switcher.get(self.algorithm, lambda: "nothing")

		dist = func(train, test)

		return dist

	def opencv_recognize_method(self, train, test):

		method = self.OPENCV_METHODS[self.algorithm]

		distance = cv2.compareHist(train, test, method)

		return distance

	def scipy_recognize_method(self, train, test):
		method = self.SCIPY_METHODS[self.algorithm]

		distance = method(train, test)

		return distance

	def prepare_data(self):

		train_data = []
		test_data = []
		train_labels = []
		test_labels = []

		# Separate trainig and testing data
		test_images = Image.get_test_data()
		train_images = Image.get_train_data(test_images)

		# Create training data set
		for image in train_images:
			histogram_model = Histogram.get_by_image_params(image.id, self.points, self.radius, self.method)

			if histogram_model is None:
				histogram_results = HistogramMaker.create_histogram_from_b64(image.image)

				histogram_json = json.dumps(histogram_results['histogram'].tolist())

				histogram_model = Histogram(image_id=image.id,
											user_id=image.user_id,
											histogram=histogram_json,
											number_points=histogram_results['points'],
											radius=histogram_results['radius'],
											method=histogram_results['method'],
											)

				histogram_model.save()

			train_labels.append(histogram_model.user_id)
			train_data.append(numpy.asarray(json.loads(histogram_model.histogram)))

		# Create test data set

		for image in test_images:
			# print("Image user id: ", image.user_id)
			histogram_model = Histogram.get_by_image_params(image.id, self.points, self.radius, self.method)

			if histogram_model is None:
				histogram_results = HistogramMaker.create_histogram_from_b64(image.image)

				histogram_json = json.dumps(histogram_results['histogram'].tolist())

				histogram_model = Histogram(image_id=image.id,
											user_id=image.user_id,
											histogram=histogram_json,
											number_points=histogram_results['points'],
											radius=histogram_results['radius'],
											method=histogram_results['method'],
											)

				histogram_model.save()

			test_labels.append(histogram_model.user_id)
			test_data.append(numpy.asarray(json.loads(histogram_model.histogram)))

		# print(test_labels)
		# Train data size check
		train_data, test_data = self.resize_data(train_data, test_data)
		# Test data size check
		# test_data = self.resize_data(test_data)

		return train_data, train_labels, test_data, test_labels

	def resize_data(self, data, test_data):

		max_array_size = 0

		for d in data:
			if max_array_size < len(d):
				max_array_size = len(d)

		for d in test_data:
			if max_array_size < len(d):
				max_array_size = len(d)

		for idx, d in enumerate(data):
			while len(d) != max_array_size:
				d = numpy.hstack((d, d.mean()))
				data[idx] = d

		for idx, d in enumerate(test_data):
			while len(d) != max_array_size:
				d = numpy.hstack((d, d.mean()))
				test_data[idx] = d

		return data, test_data
