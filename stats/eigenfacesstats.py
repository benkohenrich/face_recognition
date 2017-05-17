from collections import Counter
from operator import itemgetter

import cv2
import numpy
from decimal import Decimal
from flask import json, current_app
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from helpers.eigenfaceshelper import EigenfacesHelper
from helpers.imagehelper import ImageHelper
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser, ResponseParser
from helpers.recognizerhelper import RecognizeHelper
from models.histogram import Histogram
from models.image import Image


class EigenfacesStats:
	SCIPY_METHODS = {
		"euclidean": dist.euclidean,
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, number_components=24, method='randomized', algorithm='chi-squared', whiten = True):
		self.method = method
		self.number_components = number_components
		self.algorithm = algorithm
		if whiten is None:
			self.whiten = True
		else:
			if whiten == 'true' or whiten == 1:
				self.whiten = True
			elif whiten == 'false' or whiten == 0:
				self.whiten = False
			else:
				self.whiten = True

	def check_distances(self):

		score = []
		y = []
		for x in range(0, 10):
			print("Prepare ", x)
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
			print("Cross ", x)
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

					found_ID = min(distances)[1]
					distance = min(distances)[0]

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

		filename = 'PCA ROC curve: method: ' + str(self.method)

		plt.figure()
		label = "AUC = %0.2f\n" % roc_auc
		label += "Algorithm = %s " % self.algorithm
		plt.plot(newFPR, newTPR, label=label)
		plt.title(filename)
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		# filename = 'statistics/pca-roc-a-' + self.algorithm + '-method-' + self.method + '.png'
		image_id = ImageHelper.save_plot_image(plt, 'roc', g.user.id)
		ResponseParser().add_image('roc', 'roc_graph', image_id)


	def check3(self):

		true_positive_rate = []
		false_positive_rate = []

		TPR_m = []
		FPR_m = []

		roc_auc_max = 0

		for x in range(0, 10):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			# print(train_data)
			# print(test_data)
			train_data, test_data = RecognizeHelper.normalize_data(train_data, test_data)

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
				thr = numpy.linspace(min_score, max_score, 1000)
				FP = 0
				TP = 0
				FN = 0
				TN = 0
				print(train_labels)
				# N = 8
				# P = len(y) - N

				for (h, T) in enumerate(thr):
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
				if roc_auc > 0.50:
					FPR_m.append(roc_x)
					TPR_m.append(roc_y)

				print("AUC:", roc_auc)

				if roc_auc > roc_auc_max:
					roc_auc_max = roc_auc
					true_positive_rate = roc_y
					false_positive_rate = roc_x

		newTPR = []
		newFPR = []
		for i in range(0, 1000):
			values = []
			valuesFalse = []
			for vector in TPR_m:
				values.append(vector[i])
			for vector in FPR_m:
				valuesFalse.append(vector[i])

			newTPR.append(numpy.mean(values))
			newFPR.append(numpy.mean(valuesFalse))

		# newFPR.append(0.0)
		# newTPR.append(0.0)
		# newFPR.append(1.0)
		# newTPR.append(1.0)
		newTPR = sorted(newTPR, key=float)
		newFPR = sorted(newFPR, key=float)
		print("TPR NEW:", newTPR)
		print("FPR NEW:", newFPR)
		roc_auc = auc(newFPR, newTPR)

		print("AUC New:", roc_auc)
		# print("TPR:", true_positive_rate)
		# print("FPR", false_positive_rate)
		print("AUC MAX", roc_auc_max)
		# return
		plt.figure()
		plt.plot(false_positive_rate, true_positive_rate)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		ImageHelper.save_plot_image(plt, 'test', 10)

	def check2(self):

		true_positive_rate = []
		false_positive_rate = []

		TPR_m = []
		FPR_m = []

		roc_auc_max = 0

		for x in range(0, 10):
			train_data, train_labels, test_data, test_labels = self.prepare_data()

			# print(train_data)
			# print(test_data)
			train_data, test_data = RecognizeHelper.normalize_data(train_data, test_data)

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
				thr = numpy.linspace(min_score, max_score, 100)
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
				if roc_auc > 0.50:
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
		for i in range(0, 100):
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
		newFPR.append(1.0)
		newTPR.append(1.0)
		newTPR = sorted(newTPR, key=float)
		newFPR = sorted(newFPR, key=float)
		print("TPR NEW:", newTPR)
		print("FPR NEW:", newFPR)
		roc_auc = auc(newFPR, newTPR)

		print("AUC New:", roc_auc)
		# print("TPR:", true_positive_rate)
		# print("FPR", false_positive_rate)
		print("AUC MAX", roc_auc_max)
		# return
		plt.figure()
		plt.plot(newFPR, newTPR)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		ImageHelper.save_plot_image(plt, 'roc', 10)

	def prepare_data(self):

		train_data = []
		test_data = []
		train_labels = []
		test_labels = []

		# Separate trainig and testing data
		test_images = Image.get_test_data()
		train_images = Image.get_train_data(test_images)

		total_image = len(train_images)

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = numpy.zeros([total_image, current_app.config['IMG_RES']], dtype='float64')
		y = []
		images = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in train_images:
			xx = EigenfacesHelper.prepare_image(image.image)
			if xx is None:
				continue
			X[c, :] = xx
			train_labels.append(image.user_id)
			c += 1

		# Train data with PCA
		print("PCA train: nc=", self.number_components, " w=", self.whiten, " svd=", self.method)
		pca = PCA(n_components=self.number_components, whiten=self.whiten, svd_solver=self.method)

		train_data = pca.fit_transform(X)

		for image in test_images:
			testX = EigenfacesHelper.prepare_image(image.image)
			test_labels.append(image.user_id)
			c += 1

			t = []
			t2 = []
			for x in testX:
				t2.append(x)
			t.append(t2)
			test_data.append(pca.transform(t)[0])

		train_data, test_data = RecognizeHelper.normalize_data(train_data, test_data)

		return train_data, train_labels, test_data, test_labels

	def calculate_distance(self, test, train):

		train = train
		test = test
		# print(self.algorithm)
		switcher = {
			# "svm": self.svm_recognize,
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

	def scipy_recognize_method(self, train, test):
		method = self.SCIPY_METHODS[self.algorithm]

		distance = method(train, test)

		return distance
