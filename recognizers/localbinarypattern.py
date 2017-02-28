import argparse
import cv2
from flask import json
from flask import request
from sklearn.svm import LinearSVC

from models.base import db
from models.histogram import Histogram
from models.image import Image
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser
import numpy as np


class LBPRecognizer:
	def __init__(self, histogram_id, num_points=24, range=8, method='uniform'):
		self.points = int(num_points)
		self.range = int(range)
		self.method = str(method)
		self.input_parser = InputParser()
		self.comparing_histogram_id = histogram_id

	def recognize(self):

		argument = self.input_parser.__getattr__('algorithm')

		switcher = {
			'svm': self.svm_recognize,
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# print(func)
		# Execute the function
		func()




# display the image and the prediction
# cv2.imshow("Image", image)
# cv2.waitKey(0)
#
# print("end of calculating......")

	def svm_recognize(self):
		print("Linear Support Vector Machine ")

		data = []
		labels = []

		all_image = Image.query.all()
		total_image = Image.query.count()

		for image in all_image:
			histogram_model = Histogram.query.filter(Histogram.image_id == image.id).first()

			if histogram_model is None:
				options = {
					'range': self.range,
					'method': self.method,
					'points': self.points
				}

				histogram_results = HistogramMaker.create_histogram_from_b64(image.image)
			
				histogram_json = json.dumps(histogram_results['histogram'].tolist())
				histogram_model = Histogram(image_id=image.id,
											histogram=histogram_json,
											number_points=histogram_results['points'],
											radius=histogram_results['radius'],
											method=histogram_results['method']
											)
				histogram_model.save()

			labels.append(histogram_model.user_id)
			# todo string histogram to array
			data.append(np.asarray(json.loads(histogram_model.histogram)))

		print(data)
		print(labels)
		model = LinearSVC(C=100.0, random_state=42)
		model.fit(data, labels)

		# loop over the testing images
		# todo get the new face histogram for recognization
		hist = Histogram.query.filter(Histogram.image_id == self.comparing_histogram_id).first()
		hist = "[0.059, 0.06746666666666666, 0.0628, 0.4444, 0.15346666666666667, 0.1236, 0.06133333333333333, 0.05555, 0.07426666666666666, 0.18586666666666668]"
		# hist = json.loads(hist.histogram)
		hist = json.loads(hist)
		hist = np.asarray(hist)
		print(hist)
		print(type(hist))
		prediction = model.predict(hist.reshape(1, -1))[0]
		# prediction = model.predict(hist)[0]
		# todo log the prediction and send back result
		print(prediction)