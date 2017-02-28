import os

# from flask import request
import singleton as singleton
from flask import json
from flask import jsonify
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
	extraction_settings = None
	recognition_settings = None
	histogram = None

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
			self.parse_attributes(my_request.json)

	def parse_attributes(self, attributes):

		data = attributes

		# print(data)
		if data.get('face', None) is not None:
			self.face = data.get('face')

		if data.get('face_type', None) is not None:
			if data.get('face_type', None) in self.allowed_face_types:
				self.face_type = data.get('face_type', None)

		if data.get('extraction_settings', None) is not None:
			self.extraction_settings = data.get('extraction_settings')

		if data.get('recognition_settings', None) is not None:
			self.recognition_settings = data.get('recognition_settings')

		if data.get('histogram', None) is not None and self.face_type == "histogram":
			self.histogram = data.get('histogram')

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

		if self.extraction_settings.get(item) is not None:
			return self.extraction_settings.get(item)

		if self.recognition_settings.get(item) is not None:
			return self.recognition_settings.get(item)

		return None

class ResponseParser:
	__instance = None

	process_codes = ['extraction','db_save','recognition']

	response_data = {}

	def __new__(self):
		if not hasattr(self, 'instance'):
			self.instance = super(ResponseParser, self).__new__(self)

		return self.instance

	def add_process(self, code, data):

		if not code in self.response_data:
			self.response_data[code] = []

		self.response_data[code] = data


