import uuid

from models.process import Process as ProcessModel
from models.algorithm_type import AlgorithmType as AlgorithmTypeModel

class Process(object):
	__instance = None

	process_id = None
	process = None

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

