import traceback

from flask import request
from flask_restful import Resource, abort

from helpers.imagehelper import ImageHelper
from models.base import db
from models.user import User


class Users(Resource):
	@staticmethod
	def registration():
		try:
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

			if "base64" in avatar:
				image_path = ImageHelper.decode_base64_to_filename(avatar)
				ImageHelper.minimalize(image_path, 200)
				avatar = ImageHelper.encode_base64_from_path(image_path)
				avatar = ImageHelper.decode_base64(avatar.decode())
				full_id = ImageHelper.save_image(avatar, 'avatar', user.id)

			return user.username
		except:
			traceback.print_exc()
			db.session.rollback()
			abort(500)




