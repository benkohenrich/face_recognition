from models.algorithm_type import AlgorithmType
from models.base import Base, db
from models.process_detail import ProcessDetail


class Process(Base):
	__tablename__ = "processes"

	user_id = db.Column(db.Integer, index=True)
	algorithm_type_id = db.Column(db.Integer, index=True)
	uuid = db.Column(db.String(255), nullable=False, unique=True)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()

	def summary(self):

		algorithm_type = AlgorithmType.query.filter(AlgorithmType.id == self.algorithm_type_id).first()
		process_details = ProcessDetail.query.filter(ProcessDetail.process_id == self.id).all()

		data = {
			"user_id": self.user_id,
			"algorithm_type": algorithm_type.name,
			"uuid": self.uuid,
			"created_at": self.created_at,
		}

		return data