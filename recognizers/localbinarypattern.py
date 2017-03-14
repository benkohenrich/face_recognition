import argparse
import cv2
#Flask
import json
from flask import request
#OpenCv
from sklearn.svm import LinearSVC
#Models
from models.base import db
from models.histogram import Histogram
from models.image import Image
#Helpers
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser
from helpers.parsers import ResponseParser
import numpy as np


class LBPRecognizer:
	def __init__(self, histogram_id, num_points=24, range=8, method='uniform'):
		self.points = int(num_points)
		self.range = int(range)
		self.method = str(method)
		self.input_parser = InputParser()
		self.comparing_histogram = histogram_id

	def recognize(self):

		argument = self.input_parser.__getattr__('algorithm')

		switcher = {
			'svm': self.svm_recognize,
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()


	def svm_recognize(self):
		print("Linear Support Vector Machine ")

		data = []
		labels = []

		all_image = Image.get_all_to_extraction()

		total_image = all_image.count()

		print("########## START RECOGNITION #########")
		for image in all_image:
			histogram_model = Histogram.get_by_image_params(image.id, self.points, self.range, self.method)

			if histogram_model is None:

				histogram_results = HistogramMaker.create_histogram_from_b64(image.image)
			
				histogram_json = json.dumps(histogram_results['histogram'].tolist())

				histogram_model = Histogram(image_id=image.id,
											histogram=histogram_json,
											number_points=histogram_results['points'],
											radius=histogram_results['radius'],
											method=histogram_results['method'],
											gray_image_id=histogram_results['gray_image_id']
											)
				histogram_model.save()

			labels.append(histogram_model.user_id)
			data.append(np.asarray(json.loads(histogram_model.histogram)))

		# print(data)
		print(labels)
		model = LinearSVC(C=100.0, random_state=42)
		print("########## FIT MODEL #########")
		model.fit(data, labels)

		# hist_test = "[0.059, 0.06746666666666666, 0.0628, 0.4444, 0.15346666666666667, 0.1236, 0.06133333333333333, 0.05555, 0.07426666666666666, 0.18586666666666668]"

		hist = self.comparing_histogram

		print("########## PREDICT #########")
		prediction = model.predict(hist.reshape(1, -1))[0]
		# prediction = model.predict(hist)[0]
		print(prediction)

		process = {
			"parameters" : {
				"algorithm" : "svm",
				"recognize_histogram" : json.dumps(hist.tolist()),
				"total_compared_histograms" : total_image,
				"predict_user" : {
					"id" : int(prediction),
					"name" : "",
					"main_image" : ""
				},
			},
			"messages": {

			},
			"images": {

			},
			"metadata" : {

			}
		}

		ResponseParser().add_process('recognition', process)
		print("########## END #########")
