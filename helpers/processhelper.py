import uuid

from flask import json, jsonify

from helpers.parsers import ErrorParser, InputParser, ResponseParser
from models.process import Process as ProcessModel
from models.algorithm_type import AlgorithmType as AlgorithmTypeModel
from models.process_detail import ProcessDetail


class Process(object):
	__instance = None

	process_id = None
	process = None
	code = None

	def __new__(self):
		if not hasattr(self, 'instance'):
			self.instance = super(Process, self).__new__(self)

		return self.instance

	def create_new_process(self, user_id, algorithm):

		type = AlgorithmTypeModel.query.filter(
			AlgorithmTypeModel.code==algorithm
		).first()

		process = ProcessModel(
			user_id=user_id,
			algorithm_type_id=type.id,
			process_hash=uuid.uuid4()
		)

		process.save()

		self.process_id = process.id
		self.process = process

	def set_code(self, code):
		self.code = code

	def generate(self):

		if not ErrorParser().is_empty():

			detail = ProcessDetail(
				process_id=self.process_id,
				code='errors',
				errors=jsonify(ErrorParser().get_errors()),
				# inputs=InputParser().get_inputs()
			)

			detail.save()

		else:

			detail = ProcessDetail(
				process_id=self.process_id,
				code=self.code,
				# inputs=InputParser().get_inputs()
				responses=json.dumps((ResponseParser().get_response_data()))
			)

			if self.code == 'extraction':
				detail.extraction_settings = json.dumps((ResponseParser().get_response_data()['extraction']))
			elif self.code == 'recognition':
				try:
					detail.extraction_settings = jsonify(ResponseParser().get_response_data()['extraction'])
				except:
					detail.extraction_settings = None

				detail.recognition_settings = jsonify(ResponseParser().get_response_data()['recognition'])

			detail.save()

