from models.base import Base, db


class ProcessDetail(Base):
	__tablename__ = "process_details"

	process_id = db.Column(db.Integer, index=True)
	code = db.Column(db.String(100))
	responses = db.Column(db.TEXT, nullable=True)
	inputs = db.Column(db.TEXT, nullable=True)
	errors = db.Column(db.TEXT, nullable=True)
	extraction_settings = db.Column(db.TEXT, nullable=True)
	recognition_settings = db.Column(db.TEXT, nullable=True)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()

