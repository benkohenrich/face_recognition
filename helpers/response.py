import os

from flask import json
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

# image_content = face_file.read()
#     batch_request = [{
#         'image': {
#             #'content': base64.b64encode(image_content)
#             'content': str(base64.b64encode(image_content).decode("utf-8"))
#             },
#         'features': [{
#             'type': 'FACE_DETECTION',
#             'maxResults': max_results,
#             }]
#         }]

	# @staticmethod
	# def validate(response, d, depth=0):
	# 	for k, v in sorted(d.items(), key=lambda x: x[0]):
	# 		if isinstance(v, dict):
	# 			ResponseHelper.validate(response, v, depth + 1)
	# 		else:
	# 			if isinstance(v, str):
	# 				response
	#