from models.base import Base, db


class Process(Base):
	__tablename__ = "processes"

	user_id = db.Column(db.Integer, index=True)
	algorithm_type_id = db.Column(db.Integer, index=True)
	process_hash = db.Column(db.String(255), nullable=False, unique=True)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()

