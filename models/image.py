import base64
import enum
from models.base import Base, db

class Image(Base):

	__tablename__ = "images"

	type = db.Column(
		db.String(64),
		default=str('histogram')
	)

	user_id = db.Column(db.Integer, nullable=True)
	image = db.Column(db.BLOB, nullable=False)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()