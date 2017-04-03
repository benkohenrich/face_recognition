from flask import current_app, url_for
from sqlalchemy import func

from models.base import Base, db
from models.user import User


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
	def get_by_id(image_id):
		return Image.query.filter(Image.id == image_id).first()

	@staticmethod
	def get_all_to_extraction():
		all_image = []

		users = User.query.all()

		for user in users:
			images = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.filter() \
				.order_by(func.rand()) \
				.limit(current_app.config.get('PREPARE_PER_USER_IMAGES')) \
				.all()

			all_image += images

		return all_image

	@staticmethod
	def remove(id):
		Image.query.filter(Image.id == id).delete()
		db.session.commit()

	@staticmethod
	def avatar_path(user_id):
		avatar = Image.query.filter(Image.user_id == user_id).filter(Image.type == 'avatar').first()
		url = ''

		if avatar is not None:
			if current_app.config['URL_NAME'] is None:
				url = url_for('get_image', image_id=avatar.id)
			else:
				url = current_app.config['URL_NAME'] + url_for('get_image', image_id=avatar.id)

		return url
