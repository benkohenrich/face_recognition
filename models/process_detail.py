from flask import json

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

	def summary(self):

		if self.extraction_settings is None:
			extraction_settings = ''
		else:
			extraction_settings = json.loads(self.extraction_settings)

		if self.errors is None:
			errors = ''
		else:
			errors = json.loads(self.errors)

		data = {
			"code": self.code,
			"responses": json.loads(self.responses),
			"inputs" : json.loads(self.inputs),
			"errors": errors,
			# "extraction_settings": extraction_settings,
			# "recognition_settings": json.loads(self.recognition_settings),
		}

		return data

