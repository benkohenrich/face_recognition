import os
# import cups
import sys

import flask
from flask import Flask
from flask import g
from flask import json
from flask import request
from flask_restful import Api
from requests import auth
# from requests.auth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth

from helpers.response import ResponseHelper
from models.base import db
from models.user import User
from resources.lbp import LBPHistogram
from helpers.parsers import InputParser


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

	@app.route('/lbp/face/', methods=['GET', 'POST'])
	def lbph_photo():

		inputs = InputParser()
		inputs.set_attributes(request)

		if request.method == 'POST':

			LBPHistogram.save_histogram()

			return ResponseHelper.create_response() , 201
			# return ResponseHelper.create_response(response, message), 200
		else:
			return ResponseHelper.create_response(message), 200

	@app.route('/lbp/', methods=['POST'])
	def lbp():

		inputs = InputParser()
		inputs.set_attributes(request)

		LBPHistogram.recognize_face()

		return ResponseHelper.create_response(message), 200

	@app.route("/")
	def hello():
		return "Hello World!"

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


	# api.add_resource(LBPHistogram, '/lbph/histogram/', 'post')
	# api.add_resource(LBPHistogram, '/lbph/', methods=['GET', 'POST'])
	# api.add_resource(LBPHistogram, '/xml/', endpoint='xml', strict_slashes=False)

	return app
