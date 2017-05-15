import base64

import binascii
from flask import json, current_app, url_for, abort


class Test(object):
	__instance = None
	attr = "empty"

	def __new__(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Test, cls).__new__(cls)

		return cls.instance

	def reset(self):
		self.__instance = None


class InputParser(object):
	http_method = None

	face = None
	face_type = None
	extraction_settings = {}
	recognition_settings = {}
	histogram = None
	stats_type = None

	validate_attributes = {}

	allowed_face_types = ["full", "full_grey", "face", "face_grey", "histogram"]

	is_recognize = False

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

		if 'stats' not in self.validate_attributes:
			if data.get('face_type', None) is not None:
				if data.get('face_type', None) in self.allowed_face_types:
					self.face_type = data.get('face_type', None)
				else:
					errors.add_error('face_type', 'Face type is not_allowed: ' + data.get('face_type', None))
			else:
				errors.add_error('face_type', 'generals.face_type.required')

			if data.get('face', None) is not None and data.get('face', None) is not '':

				try:
					imgdata = base64.b64decode(data.get('face', None))
				except binascii.Error:
					ErrorParser().add_error('face', 'Face has bad format!')
					abort(422)

				self.face = data.get('face')
			elif data.get('histogram', None) is not None:
				self.histogram = data.get('histogram')
			else:
				errors.add_error('face', 'Image or histogram is required')

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

		if data.get('stats_type') is not None:
			self.stats_type = data.get('stats_type')

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

	def get_inputs(self):
		result = {
			'face': self.face,
			'face_type': self.face_type,
			'extraction_settings': self.extraction_settings,
			'recognition_settings': self.recognition_settings,
			'histogram': self.histogram,
			'stats_type': self.stats_type
		}

		return result

	def reset(self):
		self.http_method = None

		self.face = None
		self.face_type = None
		self.extraction_settings = {}
		self.recognition_settings = {}
		self.histogram = None
		self.validate_attributes = {}
		self.is_recognize = False


class ResponseParser:
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

		if current_app.config['URL_NAME'] is None:
			url = url_for('get_image', image_id=image_id)
		else:
			url = current_app.config['URL_NAME'] + url_for('get_image', image_id=image_id)

		if type == 'extraction':
			self.extraction_images[code] = url
		else:
			self.recognition_images[code] = url

	def get_response_data(self):

		try:
			self.response_data['extraction']['images'] = self.extraction_images
		except KeyError:
			print('no extraction')

		try:
			self.response_data['recognition']['images'] = self.recognition_images
		except KeyError:
			print('no extraction')

		return self.response_data

	def reset(self):
		self.response_data = {}
		self.extraction_images = {}
		self.recognition_images = {}


class ErrorParser:
	_errors = {}

	def __new__(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = super(ErrorParser, cls).__new__(cls)

		return cls.instance

	def is_empty(self):
		return len(self._errors) == 0

	def add_error(self, code, message):
		self._errors[code] = message

	def get_errors(self):
		return self._errors

	def reset(self):
		self._errors = {}
