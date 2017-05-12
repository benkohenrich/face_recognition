from flask import g
from flask_restful import Resource
from helpers.imagehelper import ImageHelper
from helpers.processhelper import Process
from recognizers.eigenfaces import EigenfacesRecognizer

from helpers.parsers import InputParser, ErrorParser, ResponseParser
from recognizers.fisherfaces import FisherfacesRecognizer


class Fisherfaces(Resource):
	@staticmethod
	def recognize_face():

		print("Fisherfaces recognizer")
		Fisherfaces.validate_attributes('recognition')
		if not ErrorParser().is_empty():
			return

		face, parent_id = ImageHelper.prepare_face_new(InputParser().face, InputParser().face_type)
		if face is None:
			return

		if Process().is_new:
			image_id = ImageHelper.save_image(face, 'face', g.user.id, parent_id)
			ResponseParser().add_image('extraction', 'face', image_id)
			Process().face_image_id = image_id

		face = ImageHelper.encode_base64(face)

		number_components = InputParser().__getattr__('number_components')

		if  number_components is not None:
			if  number_components == 0 or number_components == '0':
				number_components = None
			else:
				number_components = int(number_components)

		recognizer = FisherfacesRecognizer(face, number_components, InputParser().__getattr__('tolerance'))
		recognizer.recognize()

	@staticmethod
	def validate_attributes(type='normal'):

		errors = ErrorParser()

		# if InputParser().__getattr__('number_components') is None:
		# 	errors.add_error('number_components', 'extraction.number_components.required')

		if InputParser().__getattr__('tolerance') is None:
			errors.add_error('method', 'extraction.tolerance.required')

		if type == 'recognition':
			if InputParser().__getattr__('algorithm') is None:
				errors.add_error('algorithm', 'recognition.algorithm.required')

			print(InputParser().__getattr__('algorithm'))
			if InputParser().__getattr__('algorithm') not in {'svm', 'euclidean', "manhattan", "chebysev", "cosine",
															  "braycurtis"}:
				errors.add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

		return errors
