import cv2

from flask import g
from flask import json
from flask_restful import Resource, abort
from flask import request

# from requests import api
from packages.lbph1.pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths

from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser, ResponseParser, ErrorParser
from helpers.lbphelper import HistogramMaker
from helpers.response import ResponseHelper

from models.base import db
from models.image import Image
from models.histogram import Histogram

from recognizers.localbinarypattern import LBPRecognizer
import numpy as np

CLASSIFICATION_ALGORITHM = {
	"svm",
	"correlation",
	"chi-squared",
	"intersection",
	"bhattacharyya",
	"euclidean",
	"manhattan",
	"chebysev",
	"cosine",
	"braycurtis",
}


class LBPHistogram(Resource):
	@staticmethod
	def recognize_face():
		i_parser = InputParser()
		i_parser.is_recognize = True
		face = i_parser.face
		histogram = i_parser.histogram

		# Validate parameters
		errors = LBPHistogram.validate_attributes()

		if not errors.is_empty():
			return

		if face is not None:

			face = ImageHelper.prepare_face(face, i_parser.face_type)
			# Save image to DB
			# image_id = ImageHelper.save_image(face, 'face', g.user.id)
			# ResponseParser().add_image('extraction', 'face', image_id)

			histogram_id = HistogramMaker.create_histogram_from_b64(face)

			recognizer = LBPRecognizer(
				histogram_id.get('histogram'),
				i_parser.__getattr__('points'),
				i_parser.__getattr__('radius'),
				i_parser.__getattr__('method')
			)

			recognizer.recognize()
		elif histogram is not None:
			try:
				histogram = np.asarray(json.loads(histogram))
			except:
				ErrorParser().add_error('histogram_format', 'recognize.histogram.invalid_format')
				return

			recognizer = LBPRecognizer(
				histogram,
				i_parser.__getattr__('points'),
				i_parser.__getattr__('radius'),
				i_parser.__getattr__('method')
			)

			recognizer.recognize()

	@staticmethod
	def save_histogram():

		face = InputParser().face
		histogram = InputParser().histogram

		# Validate parameters
		errors = LBPHistogram.validate_attributes()
		if not errors.is_empty():
			return

		# Save histogram from face
		if face is not None:
			image = ImageHelper.prepare_face(face, InputParser().face_type)

			# Save image to DB
			image_id = ImageHelper.save_image(image, 'face', g.user.id)
			ResponseParser().add_image('extraction', 'face', image_id)

			# Create histogram
			histogram_results = HistogramMaker.create_histogram_from_b64(image)

			histogram_json = json.dumps(histogram_results['histogram'].tolist())
			histogram_results['histogram'] = histogram_json

			# Save generated histogram to DB
			histogram_model = Histogram(
				image_id=image_id,
				user_id=g.user.id,
				histogram=histogram_json,
				number_points=histogram_results['points'],
				radius=histogram_results['radius'],
				method=histogram_results['method'],
			)

			histogram_model.save()

			return histogram_model.id

		elif histogram is not None:
			# Save histogram from histogram
			histogram_model = Histogram(
				user_id=g.user.id,
				histogram=histogram,
				number_points=InputParser().__getattr__('points'),
				radius=InputParser().__getattr__('radius'),
				method=InputParser().__getattr__('method'),
			)

			histogram_model.save()

			return histogram_model.id

	@staticmethod
	def validate_attributes(type='normal'):

		errors = ErrorParser()

		if InputParser().__getattr__('points') is None:
			errors.add_error('points', 'extraction.points.required')

		if InputParser().__getattr__('radius') is None:
			errors.add_error('radius', 'extraction.radius.required')

		if InputParser().__getattr__('method') is None:
			errors.add_error('method', 'extraction.method.required')
		else:
			if InputParser().__getattr__('method') not in {'default', 'ror', 'uniform', 'nri_uniform', 'var'}:
				errors.add_error('method_allowed', 'extraction.method.not_allowed')

		if type == 'recognition':
			if InputParser().__getattr__('algorithm') is None:
				errors.add_error('algorithm', 'recognition.algorithm.required')

			if InputParser().__getattr__('algorithm') not in CLASSIFICATION_ALGORITHM:
				errors.add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

		return errors
