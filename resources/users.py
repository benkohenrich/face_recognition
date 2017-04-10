import traceback

from flask import request, g, current_app, url_for
from flask_restful import Resource, abort

from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser
from models.base import db
from models.user import User


class Users(Resource):
	@staticmethod
	def registration():

			username = request.json.get('username')
			password = request.json.get('password')
			avatar = request.json.get('avatar')
			name = request.json.get('name')

			if username is None or password is None:
				abort(400)  # missing arguments
			if User.query.filter_by(username=username).first() is not None:
				abort(400)  # existing user

			user = User(username=username, name=name)
			user.hash_password(password)
			db.session.add(user)
			db.session.commit()

			try:
				if avatar is not None:
					if "base64" in avatar:
						image_path = ImageHelper.decode_base64_to_filename(avatar)
						ImageHelper.minimalize(image_path, 200)
						avatar = ImageHelper.encode_base64_from_path(image_path)
						avatar = ImageHelper.decode_base64(avatar.decode())
						full_id = ImageHelper.save_image(avatar, 'avatar', user.id)
			except:
				traceback.print_exc()
				db.session.rollback()

			return user.username

	@staticmethod
	def save_face_image():
		try:
			image = ImageHelper.prepare_face(InputParser().face, InputParser().face_type)

			# Save image to DB
			if image is None:
				return

			image_id = ImageHelper.save_image(image, 'face', g.user.id)

			if current_app.config['URL_NAME'] is None:
				url = "http://0.0.0.0:5000" + url_for('get_image', image_id=image_id)
			else:
				url = current_app.config['URL_NAME'] + url_for('get_image', image_id=image_id)

			return url
		except:
			traceback.print_exc()
			db.session.rollback()
			abort(500)




