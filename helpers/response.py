import os

from flask import json
from flask import jsonify
from helpers.parsers import ResponseParser, ErrorParser, InputParser
from helpers.processhelper import Process


class ResponseHelper(object):
	@staticmethod
	def create_response(message=""):

		errors = ErrorParser()

		response = {'metadata': {}}

		response['metadata']['estimated_time'] = 0

		if not errors.is_empty():
			response['errors'] = errors.get_errors()
		else:
			if Process().process is not None:
				response['process'] = ResponseParser().get_response_data()
				response['process']['uuid'] = Process().process.uuid
				Process().generate()

		ErrorParser().reset()
		InputParser().reset()
		ResponseParser().reset()

		return jsonify(response)