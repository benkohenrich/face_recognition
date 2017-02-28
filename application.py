import os
# import cups
import sys

import flask
from flask import Flask
from flask import json
from flask import request
from flask_restful import Api

from helpers.response import ResponseHelper
from models.base import db
from resources.lbphistogram import LBPHistogram
from helpers.parsers import InputParser


def create_app():
	app = Flask(__name__)
	app.config.from_object('config.BaseConfig')
	# api = Api(app)

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

	# api.add_resource(LBPHistogram, '/lbph/histogram/', 'post')
	# api.add_resource(LBPHistogram, '/lbph/', methods=['GET', 'POST'])
	# api.add_resource(LBPHistogram, '/xml/', endpoint='xml', strict_slashes=False)

	return app
