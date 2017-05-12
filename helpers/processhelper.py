import uuid

from flask import json, jsonify, request

from helpers.parsers import ErrorParser, InputParser, ResponseParser
from models.process import Process as ProcessModel
from models.algorithm_type import AlgorithmType as AlgorithmTypeModel
from models.process_detail import ProcessDetail


class Process(object):
	__instance = None

	process_id = None
	process = None
	code = None

	is_new = True
	face_image_id = None

	def __new__(cls):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Process, cls).__new__(cls)

		return cls.instance

	def create_new_process(self, user_id, algorithm):

		type = AlgorithmTypeModel.query.filter(
			AlgorithmTypeModel.code==algorithm
		).first()

		try:
			header_uuid = request.headers['process-uuid']
		except:
			header_uuid = None

		make_new = True

		if header_uuid is not None:
			process = ProcessModel.query().filter(ProcessModel.uuid == header_uuid).first()

			if process is not None:
				make_new = False
				is_new = False

		if make_new:
			process = ProcessModel(
				user_id=user_id,
				algorithm_type_id=type.id,
				uuid=uuid.uuid4()
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
			)
			detail.save()
		else:
			detail = ProcessDetail(
				process_id=self.process_id,
				code=self.code,
				inputs=json.dumps(InputParser().get_inputs()),
				responses=json.dumps((ResponseParser().get_response_data()))
			)

			if self.code == 'extraction':
				detail.extraction_settings = json.dumps((ResponseParser().get_response_data()['extraction']))
			elif self.code == 'recognition':
				try:
					detail.extraction_settings = json.dumps((ResponseParser().get_response_data()['extraction']))
				except:
					detail.extraction_settings = None

				try:
					detail.recognition_settings = json.dumps((ResponseParser().get_response_data()['recognition']))
				except:
					detail.recognition_settings = None

			detail.save()

