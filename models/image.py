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
	parent_id = db.Column(db.Integer, nullable=True)
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
	def get_all_to_extraction(actual_face_id=None):
		all_image = []

		users = User.query.all()
		print(actual_face_id)
		for user in users:
			images = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.filter(Image.id != actual_face_id) \
				.order_by(func.rand()) \
				.limit(current_app.config.get('PREPARE_PER_USER_IMAGES')) \
				.all()

			all_image += images

		return all_image

	@staticmethod
	def remove(image_id):
		Image.query.filter(Image.id == image_id).delete()
		db.session.commit()

	@staticmethod
	def remove_by_parent(parent_id):
		try:
			Image.query.filter(Image.parent_id == parent_id).delete()
			db.session.commit()
		except:
			return

	@staticmethod
	def avatar_path(user_id):
		if user_id is None:
			return ''

		avatar = Image.query.filter(Image.user_id == user_id).filter(Image.type == 'avatar').first()

		url = ''
		if avatar is not None:
			if current_app.config['URL_NAME'] is None:
				url = url_for('get_image', image_id=avatar.id)
			else:
				url = current_app.config['URL_NAME'] + url_for('get_image', image_id=avatar.id)

		return url

	@staticmethod
	def avatar_id(user_id):
		avatar = Image.query.filter(Image.user_id == user_id).filter(Image.type == 'avatar').first()

		if avatar is None:
			return None
		else:
			return avatar.id
	@staticmethod
	def delete_avatar(user_id):
		Image.query.filter(Image.user_id == user_id).filter(Image.type == 'avatar').delete()
		db.session.commit()
		return True

	@staticmethod
	def get_train_data(test_images):
		all_image = []
		used_image_id = []
		users = User.query.all()

		for im in test_images:
			used_image_id.append(im.id)

		for user in users:
			total_image = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.count()

			if total_image < 10:
				continue

			images = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.filter(Image.id.notin_(used_image_id)) \
				.order_by(func.rand()) \
				.limit(9) \
				.all()

			all_image += images

		return all_image

	@staticmethod
	def get_test_data():
		all_image = []

		users = User.query.all()
		for user in users:
			total_image = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.count()

			if total_image < 10:
				continue

			images = Image.query \
				.filter(Image.type == 'face') \
				.filter(Image.user_id == user.id) \
				.order_by(func.rand()) \
				.limit(1) \
				.all()

			all_image += images

		return all_image

	@staticmethod
	def summary_for_user(user_id):

		result = []

		images = Image.query.filter(Image.user_id == user_id).filter(Image.type == 'face').all()

		for image in images:
			if current_app.config['URL_NAME'] is None:
				url = url_for('get_image', image_id=image.id)
			else:
				url = current_app.config['URL_NAME'] + url_for('get_image', image_id=image.id)

			result.append({
				'id': image.id,
				'url': url
			})

		return result
