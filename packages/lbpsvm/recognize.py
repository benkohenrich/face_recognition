import argparse
import cv2
from flask import json
from flask import request
from sklearn.svm import LinearSVC

from models.base import db
from models.histogram import Histogram
from models.image import Image
from helpers.lbphelper import HistogramMaker


class LBPRecognizer:
	def __init__(self, num_points=24, range=8, method='uniform'):
		self.points = int(num_points)
		self.range = int(range)
		self.method = str(method)

	def recognize(self):
		data = []
		labels = []

		all_image = Image.query.filter(Image.type == "histogram").all()
		total_image = Image.query.filter(Image.type == "histogram").count()

		for image in all_image:
			image_histogram = Histogram.query.filter(Histogram.image_id == image.id).first()

			if image_histogram is None:
				options = {
					'range': self.range,
					'method': self.method,
					'points': self.points
				}

				histogram_results = HistogramMaker.create_histogram_from_b64(image.image)
				# todo Histogram save function
				json_histogram = json.dumps(histogram_results['histogram'].tolist())
				histogram_model = Histogram(image_id=image.id, histogram=json_histogram,
											number_points=histogram_results['points'], radius=histogram_results['radius'],
											method=histogram_results['method'])
				db.session.add(histogram_model)
				db.session.commit()
				db.session.flush()

				image_histogram = histogram_model

			labels.append(image_histogram.user_id)
			# todo string histogram to array
			data.append(image_histogram.histogram)


		model = LinearSVC(C=100.0, random_state=42)
		model.fit(data, labels)

		# loop over the testing images
		# todo get the new face histogram for recognization
		hist = 1
		prediction = model.predict(hist.reshape(1, -1))[0]
# prediction = model.predict(hist)[0]
# todo log the prediction and send back result
		print(prediction)
# display the image and the prediction
# cv2.imshow("Image", image)
# cv2.waitKey(0)

print("end of calculating......")
