import os

from flask import json
from flask import jsonify
from helpers.parsers import ResponseParser, ErrorParser, InputParser
from helpers.processhelper import Process


class ResponseHelper(object):
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
			response['metadata']['estimated_time'] = 0

			if Process().process is not None:
				response['process'] = ResponseParser().get_response_data()
				response['process']['uuid'] = Process().process.uuid
				Process().generate()

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
