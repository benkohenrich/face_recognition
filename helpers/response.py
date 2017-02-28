import os
from flask import jsonify
from helpers.parsers import ResponseParser



class ResponseHelper(object):
	@staticmethod
	def create_response(message=""):
		response={}
		# response['message'] = message
		# response['data'] = data
		response['metadata'] = {}
		response['metadata']['estimated_time'] = 0

		response['process'] = ResponseParser().response_data


		print(response)
		return jsonify(response)
