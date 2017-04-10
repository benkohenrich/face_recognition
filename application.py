import flask
import time
from flask import Flask
from flask import abort
from flask import g, jsonify, request
from flask import make_response
from flask_httpauth import HTTPBasicAuth

from helpers.imagehelper import ImageHelper
from helpers.utilshelper import Utils
from models.base import db
from models.image import Image
from models.user import User

from resources.fisherface import Fisherfaces
from resources.lbp import LBPHistogram
from resources.eigenfaces import Eigenfaces
from resources.users import Users

from helpers.response import ResponseHelper
from helpers.parsers import InputParser, ErrorParser, ResponseParser, Test
from helpers.processhelper import Process


def create_app():
	app = Flask(__name__)
	app.config.from_object('config.BaseConfig')

	# Auth
	auth = HTTPBasicAuth()

	# Database init
	db.init_app(app)

	response = []
	message = []

	# Router
	@app.route("/tomi/")
	@auth.login_required
	def heno():
		return "Hello Holgye!"

	# Local Binary Pattern routers
	@app.route('/api/lbp/face/', methods=['GET', 'POST'])
	@auth.login_required
	def lbp_face():
		Utils.reset_singletons()
		# CREATE NEW PROCESS
		Process().create_new_process(g.user.id, 'lbp')
		Process().set_code('extraction')

		# PARSE INPUTS
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
		Utils.reset_singletons()
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

	@app.route('/api/eigen/', methods=['POST'])
	@auth.login_required
	def eigenfaces():
		Utils.reset_singletons()
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
		Utils.reset_singletons()
		# CREATE NEW PROCESS
		Process().create_new_process(g.user.id, 'eigenfaces')
		Process().set_code('recognition')

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
		Utils.reset_singletons()
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

	@app.route('/api/users/face/', methods=['POST'])
	@auth.login_required
	def save_image_for_user():
		Utils.reset_singletons()
		inputs = InputParser()
		inputs.set_attributes(request)

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(), 400

		url = Users.save_face_image()

		if not ErrorParser().is_empty():
			db.session.rollback()
			return ResponseHelper.create_response(), 400

		return jsonify({
			'image_face': url
		}), 201

	@app.route('/api/users/', methods=['POST'])
	def new_user():
		Utils.reset_singletons()
		username = Users.registration()
		return jsonify({'username': username}), 201

	@app.route("/images/<image_id>/", methods=['GET'])
	def get_image(image_id):
		Utils.reset_singletons()
		image = Image.get_by_id(image_id)

		if image is None:
			abort(404)

		response = make_response(image.image)
		response.headers['Content-Type'] = 'image/jpeg'
		return response

	@app.route("/api/singleton/<attr>/<seconds>/")
	def test(attr,seconds):
		Utils.reset_singletons()
		if not ErrorParser().is_empty():
			exit(400)

		print("Request: ", attr, " No set: ", Test().attr)

		Test().attr = attr
		print("Request: ", attr, " Set: ", Test().attr)
		time.sleep(int(seconds))
		print("Request: ", attr, " After wait: ", Test().attr)

		f = Test().attr
		Test().reset()
		return "Singleton test: " + f


	@app.route("/api/hidden/image/save/")
	def hidden():
		ID = '10'
		import glob
		for filename in glob.glob('static/faces/' + ID + '/*.jpg'):
			base64 = ImageHelper.encode_base64_from_path(filename)

			face = ImageHelper.prepare_face(base64.decode("utf-8"), 'full')
			# print(face)
			image_id = ImageHelper.save_image(face, 'face', 10)

		return "Done"

	@app.route("/")
	def hello():
		return "Hello World! Jakub funguje to"

	return app
