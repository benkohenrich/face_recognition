import os

# from flask import request
import singleton as singleton
from flask import json
from flask import jsonify
from flask import url_for
from singleton.singleton import Singleton


# class Singleton:
# 	def __init__(self, decorated):
# 		self._decorated = decorated
#
# 	def Instance(self):
# 		"""
#         Returns the singleton instance. Upon its first call, it creates a
#         new instance of the decorated class and calls its `__init__` method.
#         On all subsequent calls, the already created instance is returned.
#
#         """
# 		try:
# 			return self._instance
# 		except AttributeError:
# 			self._instance = self._decorated()
# 			return self._instance
#
# 	def __call__(self):
# 		raise TypeError('Singletons must be accessed through `Instance()`.')
#
# 	def __instancecheck__(self, inst):
# 		return isinstance(inst, self._decorated)


# @Singleton
class InputParser(object):
	__instance = None

	http_method = None

	face = None
	face_type = None
	extraction_settings = {}
	recognition_settings = {}
	histogram = None

	validate_attributes = {}

	allowed_face_types = ["full", "full_grey", "face", "face_grey", "histogram"]

	def __new__(self):
		if not hasattr(self, 'instance'):
			self.instance = super(InputParser, self).__new__(self)

		return self.instance

	def set_attributes(self, my_request):
		self.http_method = my_request.method

		if self.http_method == 'GET':
			if my_request.args is not None:
				self.parse_get_attributes(my_request.args)

		if self.http_method == 'POST':
			self.parse_post_attributes(my_request.json)

	def parse_post_attributes(self, attributes):

		data = attributes
		errors = ErrorParser()

		if data.get('face_type', None) is not None:
			if data.get('face_type', None) in self.allowed_face_types:
				self.face_type = data.get('face_type', None)
			else:
				errors.add_error('face_type', 'generals.face_type.not_allowed')
		else:
			errors.add_error('face_type', 'generals.face_type.required')

		if data.get('face', None) is not None and (
						data.get('face_type', None) is not 'histogram' or data.get('face_type', None) is not None):
			self.face = data.get('face')
		else:
			if data.get('face_type', None) is not 'histogram':
				errors.add_error('face', 'generals.face.required')

		if 'extraction_settings' in self.validate_attributes:
			if data.get('extraction_settings', None) is not None:
				self.extraction_settings = data.get('extraction_settings')
			else:
				errors.add_error('extraction_settings', 'extraction.required')

		if 'recognition_settings' in self.validate_attributes:
			if data.get('recognition_settings', None) is not None:
				self.recognition_settings = data.get('recognition_settings')
			else:
				errors.add_error('recognition_settings', 'recognition_settings.required')

		if data.get('histogram', None) is not None and self.face_type == "histogram":
			self.histogram = data.get('histogram')
		elif self.face_type == 'histogram':
			errors.add_error('histogram', 'generals.histogram.required')

	def parse_get_attributes(self, args):

		if args.get('face') is not None:
			self.face = args.get('face')

	# if data['face_type'] is not None:
	# 	if data['face_type'] in self.allowed_face_types:
	# 		self.face_type = data['face_type']
	#
	# if data['extraction_settings'] is not None:
	# 	self.extraction_settings = data['extraction_settings']
	#
	# if data['recognition_settings'] is not None:
	# 	self.recognition_settings = data['recognition_settings']
	#
	# if data['histogram'] is not None and self.face_type == "histogram":
	# 	self.histogram = data['histogram']

	def __getattr__(self, item):

		if self.extraction_settings.get(item, None) is not None:
			return self.extraction_settings.get(item)

		if self.recognition_settings.get(item, None) is not None:
			return self.recognition_settings.get(item)

		return None


class ResponseParser:
	__instance = None

	process_codes = ['extraction', 'db_save', 'recognition']

	response_data = {}

	extraction_images = {}
	recognition_images = {}


	def __new__(self):
		if not hasattr(self, 'instance'):
			self.instance = super(ResponseParser, self).__new__(self)

		return self.instance

	def add_process(self, code, data):

		if not code in self.response_data:
			self.response_data[code] = []

		self.response_data[code] = data

	def add_image(self, type, code, image_id):

		url = url_for('get_image', image_id=image_id)

		if type == 'extraction':
			self.extraction_images[code] = url
		else:
			self.recognition_images[code] = url

	def get_response_data(self):

		try:
			self.response_data['extraction']['images'] = self.extraction_images
		except KeyError:
			print('no extracton')

		try:
			self.response_data['recognition']['images'] = self.recognition_images
		except KeyError:
			print('no extracton')

		return self.response_data

class ErrorParser:
	__instance = None

	_errors = {}

	def __new__(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = super(ErrorParser, cls).__new__(cls)

		return cls.instance

	def is_empty(self):
		return not bool(self._errors)

	def add_error(self, code, message):
		self._errors[code] = message

	def get_errors(self):
		return self._errors
