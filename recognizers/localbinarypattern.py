import cv2

# Flask
import json
from flask import g
# OpenCv
from sklearn.svm import LinearSVC
import numpy as np
from scipy.spatial import distance as dist
# Models
from models.histogram import Histogram
from models.image import Image
from models.user import User
# Helpers
from helpers.recognizerhelper import RecognizeHelper
from helpers.utilshelper import Utils
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser, ErrorParser
from helpers.parsers import ResponseParser


class LBPRecognizer:
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

	def __init__(self, histogram_id, num_points=24, range=8, method='uniform'):
		self.points = int(num_points)
		self.range = int(range)
		self.method = str(method)
		self.input_parser = InputParser()
		self.comparing_histogram = histogram_id
		self.algorithm = "none"

	def recognize(self):

		argument = self.input_parser.__getattr__('algorithm')
		self.algorithm = argument

		switcher = {
			"svm": self.svm_recognize,
			"correlation": self.recognize_method,
			"chi-squared": self.recognize_method,
			"intersection": self.recognize_method,
			"bhattacharyya": self.recognize_method,
			"euclidean": self.scipy_recognize_method,
			"manhattan": self.scipy_recognize_method,
			"chebysev": self.scipy_recognize_method,
			"cosine": self.scipy_recognize_method,
			"braycurtis": self.scipy_recognize_method,
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()

	def scipy_recognize_method(self):

		if self.SCIPY_METHODS[self.algorithm] is None:
			ErrorParser().add_error('algorithm', '')
			return
		else:
			method = self.SCIPY_METHODS[self.algorithm]

		print("METHOD UTILIZING SCIPY")
		print("initialize the scipy methods to compaute distances")
		print("Method: ", self.algorithm)

		data, labels, total_image, image_id = self.separation()

		distances = []

		print("#### Start computing distances ####")
		for j, hist_train in enumerate(data):
			dist = method(self.np_hist_to_cv(hist_train), self.np_hist_to_cv(self.comparing_histogram))
			distances.append((dist, labels[j], image_id[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		image_ID = min(distances)[2]

		percentage = RecognizeHelper.calculate_percentage_for_distance_metric_methods(g.user.id, distance, distances)

		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ") - ", percentage, "%")

		predict_user = User.query.filter(User.id == found_ID).first()

		process = {
			"parameters": {
				'radius': self.range,
				'points': self.points,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_histogram": json.dumps(self.comparing_histogram.tolist()),
				"total_compared_histograms": total_image,
				'distance': str(distance),
				'similarity_percentage': percentage,
				"predict_user": {
					"id": int(found_ID),
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)
		ResponseParser().add_image('recognition', 'predict_image', image_ID)

	def recognize_method(self):

		reverse = False
		# if we are using the correlation or intersection
		# method, then sort the results in reverse order
		if self.algorithm in ("correlation", "intersection", "bhattacharyya"):
			reverse = True

		if self.OPENCV_METHODS[self.algorithm] is None:
			ErrorParser().add_error('algorithm', '')
			return
		else:
			method = self.OPENCV_METHODS[self.algorithm]

		print("Method: ", self.algorithm)

		data, labels, total_image, image_id = self.separation()

		distances = []

		for j, hist_train in enumerate(data):
			dist = cv2.compareHist(self.np_hist_to_cv(hist_train), self.np_hist_to_cv(self.comparing_histogram), method)
			distances.append((dist, labels[j], image_id[j]))

		#print(distances)
		if not reverse:
			found_ID = min(distances)[1]
			distance = min(distances)[0]
			image_ID = min(distances)[2]
			#print(min(distances))
		# Utils.calculate_percentage_from_distances(distances, distance)
		else:
			found_ID = max(distances)[1]
			distance = max(distances)[0]
			image_ID = max(distances)[2]
			print(max(distances))

		percentage = RecognizeHelper.calculate_percentage_for_opencv_methods(self.algorithm, distance, reverse)
		print("Identified " + self.algorithm + "(result: " + str(found_ID) + " - dist - " + repr(
			distance) + ") -  Percentage: ", percentage, "%")

		predict_user = User.query.filter(User.id == found_ID).first()

		process = {
			"parameters": {
				'radius': self.range,
				'points': self.points,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_histogram": json.dumps(self.comparing_histogram.tolist()),
				"total_compared_histograms": total_image,
				'distance': repr(distance),
				'similarity_percentage': percentage,
				"predict_user": {
					"id": int(found_ID),
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)
		ResponseParser().add_image('recognition', 'predict_image', image_ID)

	def svm_recognize(self):
		print("Linear Support Vector Machine ")

		data, labels, total_image, image_id = self.separation()

		set = Utils.remove_duplicates(labels)

		hist = self.comparing_histogram

		if len(set) == 1:
			prediction = set[0]
		# image_ID = image_id[0]
		else:
			model = LinearSVC(C=1.0, random_state=42)

		#	print(labels.__len__())

			# for d in data:
			# 	print(d.__len__())

			print("########## FIT MODEL #########")
			model.fit(data, labels)

			print("########## PREDICT #########")
			prediction = model.predict(hist.reshape(1, -1))[0]

		print(prediction)
		predict_user = User.query.filter(User.id == prediction).first()
		process = {
			"parameters": {
				"algorithm": "svm",
				"recognize_histogram": json.dumps(self.comparing_histogram.tolist()),
				"total_compared_histograms": total_image,
				"predict_user": {
					"id": int(prediction),
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
		}

		ResponseParser().add_process('recognition', process)
		# ResponseParser().add_image('recognition', 'predict_image', image_ID)
		print("########## END #########")

	def separation(self):
		print("#### Start cross validating ####")
		data = []
		labels = []
		image_id = []

		all_image = Image.get_all_to_extraction()
		total_image = len(all_image)

		for image in all_image:
			histogram_model = Histogram.get_by_image_params(image.id, self.points, self.range, self.method)

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

			image_id.append(image.id)
			labels.append(histogram_model.user_id)
			data.append(np.asarray(json.loads(histogram_model.histogram)))

		# DATA size check
		data = self.resize_data(data)

		# for d in data:
		# 	print(len(d))

		return data, labels, total_image, image_id

	def np_hist_to_cv(self, np_histogram_output):
		counts = np_histogram_output
		return_value = counts.ravel().astype('float32')
		return return_value

	def resize_data(self, data):

		max_array_size = 0
		for d in data:
			if max_array_size < len(d):
				max_array_size = len(d)

		if len(self.comparing_histogram) > max_array_size:
			max_array_size = len(self.comparing_histogram)
		elif len(self.comparing_histogram) < max_array_size:
			while len(self.comparing_histogram) != max_array_size:
				self.comparing_histogram = np.hstack((self.comparing_histogram, self.comparing_histogram.mean()))

		for idx, d in enumerate(data):
			while len(d) != max_array_size:
				d = np.hstack((d, d.mean()))
				data[idx] = d

		return data
