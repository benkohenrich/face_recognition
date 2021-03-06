import numpy as np
from flask import g
from flask import json
from flask_restful import Resource

from helpers.imagehelper import ImageHelper
from helpers.lbphelper import HistogramMaker
from helpers.parsers import InputParser, ResponseParser, ErrorParser
from helpers.processhelper import Process
from helpers.processhelper import Process as ProcessHelper
from models.histogram import Histogram
from recognizers.localbinarypattern import LBPRecognizer

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
		# Initialize
		i_parser = InputParser()
		i_parser.is_recognize = True
		face = i_parser.face
		histogram = i_parser.histogram
		is_new_process = Process().is_new
		image_id = None

		# Validate parameters
		errors = LBPHistogram.validate_attributes()
		if not errors.is_empty():
			return

		if face is not None:
			face, full_image_id = ImageHelper.prepare_face_new(face, i_parser.face_type)
			if face is None:
				return

			# Save image to DB
			if Process().is_new:
				image_id = ImageHelper.save_image(face, 'face', g.user.id, full_image_id)
				ResponseParser().add_image('extraction', 'face', image_id)
				ProcessHelper().face_image_id = image_id

			# Create a histogram from image
			histogram_id = HistogramMaker.create_histogram_from_b64(face)

			recognizer = LBPRecognizer(
				histogram_id.get('histogram'),
				i_parser.__getattr__('points'),
				i_parser.__getattr__('radius'),
				i_parser.__getattr__('method'),
				image_id
			)

			Process().is_new = False
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

			Process().is_new = False
			recognizer.recognize()

		Process().is_new = is_new_process

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
			image, full_image_id = ImageHelper.prepare_face_new(face, InputParser().face_type)
			if image is None:
				return

			# Save image to DB
			image_id = ImageHelper.save_image(image, 'face', g.user.id, full_image_id)
			ResponseParser().add_image('extraction', 'face', image_id)
			ProcessHelper().face_image_id = image_id

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
			if InputParser().__getattr__('method') not in {'default', 'ror', 'uniform', 'nri_uniform'}:
				errors.add_error('method_allowed', 'extraction.method.not_allowed')

		if type == 'recognition':
			if InputParser().__getattr__('algorithm') is None:
				errors.add_error('algorithm', 'recognition.algorithm.required')

			if InputParser().__getattr__('algorithm') not in CLASSIFICATION_ALGORITHM:
				errors.add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

		return errors
