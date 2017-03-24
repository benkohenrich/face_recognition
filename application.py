import os
# import cups
import sys

import flask
from flask import Flask, abort
from flask import g, json, jsonify, request, url_for
# from flask_restful import Api
from requests import auth
# from requests.auth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth

from models.base import db
from models.user import User
from resources.fisherface import Fisherfaces
from resources.lbp import LBPHistogram

from helpers.response import ResponseHelper
from helpers.parsers import InputParser, ErrorParser
from helpers.processhelper import Process

from resources.eigenfaces import Eigenfaces


def create_app():
	app = Flask(__name__)
	app.config.from_object('config.BaseConfig')
	# api = Api(app)

	#Auth
	auth = HTTPBasicAuth()

	# Database init
	db.init_app(app)

	response = []
	message = []

	# Router
	@app.route("/heno/")
	def heno():
		return "Hello Heno!"

	# Local Binary Pattern routers
	@app.route('/api/lbp/face/', methods=['GET', 'POST'])
	@auth.login_required
	def lbp_face():

		#CREATE NEW PROCESS
		Process().create_new_process(g.user.id, 'lbp')
		Process().set_code('extraction')

		#PARSE INPUTS
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings'}
		inputs.set_attributes(request)

		error_parser = ErrorParser()


		if not error_parser.is_empty():
			return ResponseHelper.create_response(), 400

		if request.method == 'POST':

			LBPHistogram.save_histogram()

			return ResponseHelper.create_response(), 201
		else:
			return ResponseHelper.create_response(message), 200

	@app.route('/api/lbp/', methods=['POST'])
	@auth.login_required
	def lbp():
		# CREATE NEW PROCESS
		Process().create_new_process(g.user.id, 'lbp')
		Process().set_code('recognition')

		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)

		error_parser = ErrorParser()

		if not error_parser.is_empty():
			return ResponseHelper.create_response(), 400

		LBPHistogram.recognize_face()

		return ResponseHelper.create_response(message), 200

	# Eigenfaces routers
	# @app.route('/api/eigen/face/', methods=['GET', 'POST'])
	# @auth.login_required
	# def lbph_photo():
	#
	# 	inputs = InputParser()
	# 	inputs.set_attributes(request)
	#
	# 	if request.method == 'POST':
	#
	# 		LBPHistogram.save_histogram()
	#
	# 		return ResponseHelper.create_response() , 201
	# 		# return ResponseHelper.create_response(response, message), 200
	# 	else:
	# 		return ResponseHelper.create_response(message), 200

	@app.route('/api/eigen/', methods=['POST'])
	@auth.login_required
	def eigenfaces():

		# CREATE NEW PROCESS
		Process().create_new_process(g.user.id, 'eigenfaces')
		Process().set_code('recognition')

		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(), 400

		Eigenfaces.recognize_face()

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(), 400

		return ResponseHelper.create_response(message), 200

	@app.route('/api/fisher/', methods=['POST'])
	@auth.login_required
	def fisherfaces():

		# CREATE NEW PROCESS
		# Process().create_new_process(g.user.id, 'eigenfaces')
		# Process().set_code('recognition')

		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(), 400

		Fisherfaces.recognize_face()

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(), 400

		return ResponseHelper.create_response(message), 200

	# Authorization Routers
	@app.route('/api/token/', methods=['GET'])
	@auth.login_required
	def get_auth_token():
		token = g.user.generate_auth_token()
		return flask.jsonify({'token': token.decode('ascii')})

	@auth.verify_password
	def verify_password(username_or_token, password):
		# first try to authenticate by token
		user = User.verify_auth_token(username_or_token)
		if not user:
			# try to authenticate with username/password
			user = User.query.filter_by(username=username_or_token).first()
			if not user or not user.verify_password(password):
				return False
		g.user = user
		return True

	@app.route('/api/users/', methods=['POST'])
	def new_user():
		username = request.json.get('username')
		password = request.json.get('password')
		name = request.json.get('name')
		if username is None or password is None:
			abort(400)  # missing arguments
		if User.query.filter_by(username=username).first() is not None:
			abort(400)  # existing user
		user = User(username=username, name=name)
		user.hash_password(password)
		db.session.add(user)
		db.session.commit()
		return jsonify({'username': user.username}), 201

	@app.route("/images/<image_id>/", methods=['GET'])
	def get_image(image_id):
		return image_id


	@app.route("/")
	def hello():
		return "Hello World! Jakub funguje to"
	# api.add_resource(LBPHistogram, '/lbph/histogram/', 'post')
	# api.add_resource(LBPHistogram, '/lbph/', methods=['GET', 'POST'])
	# api.add_resource(LBPHistogram, '/xml/', endpoint='xml', strict_slashes=False)

	return app
