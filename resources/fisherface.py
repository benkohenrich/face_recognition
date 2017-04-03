from flask import g
from flask_restful import Resource
from helpers.imagehelper import ImageHelper
from recognizers.eigenfaces import EigenfacesRecognizer

from helpers.parsers import InputParser, ErrorParser, ResponseParser
from recognizers.fisherfaces import FisherfacesRecognizer


class Fisherfaces(Resource):
	@staticmethod
	def recognize_face():

		Fisherfaces.validate_attributes('recognition')
		if not ErrorParser().is_empty():
			return

		face = ImageHelper.prepare_face(InputParser().face, InputParser().face_type)
		if face is None:
			return

		image_id = ImageHelper.save_image(face, 'face', g.user.id)
		ResponseParser().add_image('extraction', 'face', image_id)

		face = ImageHelper.encode_base64(face)
		recognizer = FisherfacesRecognizer(face, int(InputParser().__getattr__('number_eigenfaces')) , InputParser().__getattr__('method'))
		recognizer.recognize()


	@staticmethod
	def validate_attributes(type='normal'):

		errors = ErrorParser()

		if InputParser().__getattr__('number_eigenfaces') is None:
			errors.add_error('number_eigenfaces', 'extraction.number_eigenfaces.required')

		if InputParser().__getattr__('method') is None:
			errors.add_error('method', 'extraction.method.required')
		else:
			if InputParser().__getattr__('method') not in {'auto', 'full', 'randomized'}:
				errors.add_error('method_allowed', 'extraction.method.not_allowed')

		if type == 'recognition':
			if InputParser().__getattr__('algorithm') is None:
				errors.add_error('algorithm', 'recognition.algorithm.required')

			print(InputParser().__getattr__('algorithm'))
			if InputParser().__getattr__('algorithm') not in {'svm', 'euclidian' , "manhattan", "chebysev", "cosine", "braycurtis" }:
				errors.add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

		return errors



