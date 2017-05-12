import traceback

import binascii
from flask import request, g, current_app, url_for, abort
from flask_restful import Resource

from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser, ErrorParser

from models.base import db
from models.image import Image
from models.process import Process as ProcessModel
from models.process_detail import ProcessDetail
from models.user import User


class Users(Resource):
	@staticmethod
	def registration():
		"""
		Registration function for users
		:return: 
		"""
		username = request.json.get('username')
		password = request.json.get('password')
		avatar = request.json.get('avatar')
		name = request.json.get('name')

		if username is None or username == '':
			ErrorParser().add_error('username', 'This value is required')
			abort(422)  # missing arguments
		if password is None or password == '':
			ErrorParser().add_error('password', 'This value is required')
			abort(422)  # missing arguments
		if User.query.filter_by(username=username).first() is not None:
			ErrorParser().add_error('username', 'Username is exist')
			abort(422)  # existing user

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
			db.session.rollback()

		result = user.summary()
		result['main_photo'] = Image.avatar_path(user.id)
		return result

	@staticmethod
	def save_face_image():

		try:
			image, parent_id = ImageHelper.prepare_face_new(InputParser().face, InputParser().face_type)
		except binascii.Error:
			ErrorParser().add_error('face', 'Face has bad format!')
			abort(422)

		try:
			image_id = ImageHelper.save_image(image, 'face', g.user.id, parent_id)
			# Generation URL name for image
			if current_app.config['URL_NAME'] is None:
				url = "http://0.0.0.0:5000" + url_for('get_image', image_id=image_id)
			else:
				url = current_app.config['URL_NAME'] + url_for('get_image', image_id=image_id)

			return url
		except:
			traceback.print_exc()
			db.session.rollback()
			abort(500, "Data base internal error")

	@staticmethod
	def update(user_id):
		try:

			password = request.json.get('password')
			avatar = request.json.get('avatar')
			name = request.json.get('name')

			if password is None or password == '':
				ErrorParser().add_error('password', 'This value is required')
				abort(422)  # missing arguments

			user = User.query.filter(User.id == user_id).first()

			user.name = name
			user.hash_password(password)
			db.session.add(user)
			db.session.commit()

			try:
				if avatar is not None:
					image_path = ImageHelper.decode_base64_to_filename(avatar)
					ImageHelper.minimalize(image_path, 200)
					avatar = ImageHelper.encode_base64_from_path(image_path)
					avatar = ImageHelper.decode_base64(avatar.decode())
					full_id = ImageHelper.save_image(avatar, 'avatar', user.id)
					Image.delete_avatar(user.id)
			except:
				db.session.rollback()

			result = user.summary()
			result['main_photo'] = Image.avatar_path(user_id)
			return result
		except:
			return False

	@staticmethod
	def listing():
		response = []

		users = User.query.all()
		for user in users:
			u = user.summary()
			u['main_photo'] = Image.avatar_path(user.id)
			response.append(u)

		return response

	@staticmethod
	def get(user_id):
		try:
			user = User.query.filter(User.id == user_id).first()
			user = user.summary()
			user['main_photo'] = Image.avatar_path(user_id)
			user['images'] = Image.summary_for_user(user_id)
			return user
		except:
			return False

	@staticmethod
	def me():
		try:
			user_id = g.user.id
			user = User.query.filter(User.id == user_id).first()
			user = user.summary()
			user['images'] = Image.summary_for_user(user_id)
			user['main_photo'] = Image.avatar_path(user_id)

			return user
		except:
			return False

	@staticmethod
	def logs():
		result = []
		user_id = g.user.id
		processes = ProcessModel.query.filter(ProcessModel.user_id == user_id).all()
		for process in processes:
			result.append(process.summary())

		return result

	@staticmethod
	def log_details(log_id):
		result = []
		user_id = g.user.id

		process = ProcessModel.query.\
			filter(ProcessModel.user_id == user_id).\
			filter(ProcessModel.uuid == log_id).\
			first()

		if process is None:
			return False

		details = ProcessDetail.query.filter(ProcessDetail.process_id == process.id).all()

		for detail in details:
			result.append(detail.summary())

		print(result)
		return result
