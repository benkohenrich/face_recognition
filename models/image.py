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
	process_id = db.Column(db.Integer, nullable=True)
	image = db.Column(db.BLOB, nullable=False)
	in_storage = db.Column(db.BOOLEAN, nullable=False, default=0)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()


	@staticmethod
	def get_all_to_extraction():
		all_image = Image.query \
			.filter(Image.type == 'face').filter(Image.in_storage == 1).all()

		return all_image