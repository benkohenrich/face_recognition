import os

import time
from flask import json
from flask import jsonify
from helpers.parsers import ResponseParser, ErrorParser, InputParser
from helpers.processhelper import Process


class ResponseHelper(object):
	start_time = 0

	@staticmethod
	def create_response(code=200, message="System error"):
		response = {}
		errors = ErrorParser()

		if not errors.is_empty():
			response['errors'] = errors.get_errors()
			response['code'] = code
			response['message'] = message
		else:
			response['metadata'] = {}
			response['metadata']['estimated_time'] = time.time() - ResponseHelper().start_time

			if Process().process is not None:
				response['process'] = ResponseParser().get_response_data()
				response['process']['uuid'] = Process().process.uuid
				Process().generate()

			if ResponseParser().roc_image is not None:
				response['images'] = {}
				response['images']['roc'] = ResponseParser().roc_image

		ErrorParser().reset()
		InputParser().reset()
		ResponseParser().reset()

		return jsonify(response)

	@staticmethod
	def create_simple_response(code = 200, message="OK", errors = None):
		response = {}

		if not ErrorParser().is_empty():
			response['errors'] = ErrorParser().get_errors()
			response['code'] = code
			response['message'] = message
		else:
			if errors is not None:
				response['errors'] = errors

			response['code'] = code
			response['message'] = message

		return jsonify(response)
