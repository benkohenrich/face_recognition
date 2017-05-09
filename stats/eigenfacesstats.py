from collections import Counter
from operator import itemgetter

import cv2
import numpy
from decimal import Decimal
from flask import json
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
from sklearn.metrics import roc_curve, auc

from helpers.imagehelper import ImageHelper
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser
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

	def check2(self):

		# train_data, train_labels, test_data, test_labels = self.prepare_data()

		true_positive_rate = []
		false_positive_rate = []

		TPR_m = []
		FPR_m = []

		roc_auc_max = 0

		for x in range(0, 10):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			for i, hist_test in enumerate(test_data):
				distances = []
				score = []
				y = []

				for j, hist_train in enumerate(train_data):
					distance = self.calculate_distance(hist_test, hist_train)
					distances.append((distance, train_labels[j]))
					score.append(distance)
					y.append(train_labels[j])

				roc_x = []
				roc_y = []
				min_score = min(score)
				max_score = max(score)
				thr = numpy.linspace(min_score, max_score, 500)
				FP = 0
				TP = 0
				FN = 0
				TN = 0
				print(train_labels)
				# N = 8
				# P = len(y) - N

				for (k, T) in enumerate(thr):
					for k in range(0, len(score)):
						if (score[k] > T):
							if (y[k] == test_labels[i]):
								TP = TP + 1
							else:
								FP = FP + 1
						else:
							if int(y[k]) == test_labels[i]:
								FN += 1
							else:
								TN += 1

					TPR = TP / (TP + FN)
					FPR = FP / (FP + TN)
					roc_x.append(FPR)
					roc_y.append(TPR)
					FP = 0
					TP = 0
					FN = 0
					TN = 0

				roc_auc = auc(roc_x, roc_y)
				# if roc_auc > 0.50:
				FPR_m.append(roc_x)
				TPR_m.append(roc_y)

				print("AUC:", roc_auc)

				if roc_auc > roc_auc_max:
					roc_auc_max = roc_auc
					true_positive_rate = roc_y
					false_positive_rate = roc_x
				# print(roc_y)


		newTPR = []
		newFPR = []
		for i in range(0, 500):
			values = []
			valuesFalse = []
			for vector in TPR_m:
				values.append(vector[i])
			for vector in FPR_m:
				valuesFalse.append(vector[i])

			newTPR.append(numpy.mean(values))
			newFPR.append(numpy.mean(valuesFalse))

		newFPR.append(0.0)
		newTPR.append(0.0)
		print("TPR NEW:", newTPR)
		print("FPR NEW:", newFPR)
		roc_auc = auc(newFPR, newTPR)

		print("AUC New:", roc_auc)
		# print("TPR:", true_positive_rate)
		# print("FPR", false_positive_rate)
		# print("AUC MAX", roc_auc_max)
		# return
		plt.figure()
		plt.plot(newFPR, newTPR)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		ImageHelper.save_plot_image(plt, 'test', 10)

	def check(self):

		print("#### Start cross validating ####")

		if self.algorithm in ("correlation", "intersection", "bhattacharyya"):
			reverse = True
		else:
			reverse = False

		true_positive_rate = []
		false_positive_rate = []
		TPR_m = []
		FPR_m = []

		for x in range(0, 10):
			# True positive
			TP = 0
			# True negative
			TN = 0
			# False positive
			FP = 0
			# False negative
			FN = 0

			train_data, train_labels, test_data, test_labels = self.prepare_data()
			print("Train data length: ", len(train_data))
			print("Test data length: ", len(test_data))
			print("#### Start computing distances ####")
			for i, hist_test in enumerate(test_data):

				distances = []
				distances_only = []

				for j, hist_train in enumerate(train_data):
					distance = self.calculate_distance(hist_test, hist_train)
					distances.append((distance, train_labels[j]))
					distances_only.append(distance)

				# print("PRED:",distances_only)

				for di, do in enumerate(distances_only):
					distances_only[di] = Decimal(do).quantize(Decimal('.001'))

				print("User test: ", test_labels[i])

				counter = Counter(distances_only)
				tmp = sorted(counter.items(), key=itemgetter(1), reverse=False)
				mean = float(tmp[0][0])

				for z, d in enumerate(distances):
					if reverse:
						if d[0] >= mean:
							if int(d[1]) == test_labels[i]:
								TP += 1
							else:
								FP += 1
						else:
							if int(d[1]) == test_labels[i]:
								FN += 1
							else:
								TN += 1
					else:
						if d[0] <= mean:
							if int(d[1]) == test_labels[i]:
								TP += 1
							else:
								FP += 1
						else:
							if int(d[1]) == test_labels[i]:
								FN += 1
							else:
								TN += 1

			TPR = TP / (TP + FN)
			FPR = FP / (FP + TN)
			true_positive_rate.append(TPR)
			false_positive_rate.append(FPR)

		print(true_positive_rate)
		print(false_positive_rate)

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
		train_images = Image.get_train_data()
		test_images = Image.get_test_data(train_images)

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

		print(test_labels)
		# Train data size check
		train_data = self.resize_data(train_data)
		# Test data size check
		test_data = self.resize_data(test_data)

		return train_data, train_labels, test_data, test_labels

	def resize_data(self, data):

		max_array_size = 0

		for d in data:
			if max_array_size < len(d):
				max_array_size = len(d)

		for idx, d in enumerate(data):
			while len(d) != max_array_size:
				d = numpy.hstack((d, d.mean()))
				data[idx] = d

		return data
