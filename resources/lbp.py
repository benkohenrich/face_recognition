import cv2

from flask import json
from flask_restful import Resource, abort
from flask import request
# from requests import api
from packages.lbph1.pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths

from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser, ResponseParser
from helpers.lbphelper import HistogramMaker
from helpers.response import ResponseHelper

from models.base import db
from models.image import Image
from models.histogram import Histogram
from recognizers.localbinarypattern import LBPRecognizer


class LBPHistogram(Resource):

	@staticmethod
	def recognize_face():
		i_parser = InputParser()
		face = i_parser.face

		if face is not None:

			face = ImageHelper.prepare_face(face)

			histogram_id = HistogramMaker.create_histogram_from_b64(face)

			recognizer = LBPRecognizer(histogram_id, i_parser.__getattr__('points'), i_parser.__getattr__('radius'), i_parser.__getattr__('method'))

			recognizer.recognize()
		else:
			# todo exception no image
			print("NO IMAGE")

	@staticmethod
	def save_histogram():

		face = InputParser().face
		histogram = InputParser().histogram

		if face is not None:

			image = ImageHelper.prepare_face(face)

			# Save image to DB
			image = Image(image=image, type=InputParser().face_type)
			image.save()

			histogram_results = HistogramMaker.create_histogram_from_b64(image.image)

			histogram_original = histogram_results['histogram']
			histogram_json = json.dumps(histogram_results['histogram'].tolist())

			data = {}

			histogram_results['histogram'] = histogram_json
			
			# data['parameters'] = histogram_results
			#
			# ResponseParser().add_process('extraction', data)

			# Save generated histogram to DB
			histogram_model = Histogram(image_id=image.id,
										user_id=2,
										histogram=histogram_json,
										number_points=histogram_results['points'],
										radius=histogram_results['radius'],
										method=histogram_results['method']
										)
			histogram_model.save()
			
			return histogram_model.id

		elif histogram is not None:
			print("save histogram")


