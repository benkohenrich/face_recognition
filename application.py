import os

import flask
from flask import Flask, send_from_directory
from flask import abort
from flask import g, jsonify, request
from flask import make_response
from flask_httpauth import HTTPBasicAuth

from helpers.parsers import InputParser, ErrorParser
from helpers.processhelper import Process
from helpers.response import ResponseHelper
from helpers.utilshelper import Utils
from models.base import db
from models.histogram import Histogram
from models.image import Image
from models.user import User
from resources.eigenfaces import Eigenfaces
from resources.fisherface import Fisherfaces
from resources.lbp import LBPHistogram
from resources.stats import Stats
from resources.users import Users


def create_app():
	app = Flask(__name__, static_folder="/_docs")
	app.config.from_object('config.BaseConfig')
	# Auth
	auth = HTTPBasicAuth()
	# Database init
	db.init_app(app)

	message = []

	# Local Binary Pattern routers
	@app.route('/api/lbp/face/', methods=['POST'])
	@auth.login_required
	def lbp_face():
		Utils.reset_singletons()
		# Create a new process
		Process().create_new_process(g.user.id, 'lbp')
		Process().set_code('extraction')
		# Parse inputs to helper and validate
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings'}
		inputs.set_attributes(request)
		# Check validation errors
		error_parser = ErrorParser()
		if not error_parser.is_empty():
			return ResponseHelper.create_response(422, "Validation error"), 422

		# Run current method
		if request.method == 'POST':
			LBPHistogram.save_histogram()
			return ResponseHelper.create_response(), 201

	@app.route('/api/lbp/', methods=['POST'])
	@auth.login_required
	def lbp():
		Utils.reset_singletons()
		# Create a new process
		Process().create_new_process(g.user.id, 'lbp')
		Process().set_code('recognition')
		# Parse and validate input values
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)
		# Check validation errors
		error_parser = ErrorParser()
		if not error_parser.is_empty():
			return ResponseHelper.create_response(422, "Validation error"), 422

		# Run recognition
		LBPHistogram.recognize_face()

		# Check recognition errors
		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(400, "Bad request"), 400

		return ResponseHelper.create_response(message), 200

	@app.route('/api/eigen/', methods=['POST'])
	@auth.login_required
	def eigenfaces():
		Utils.reset_singletons()
		# Create new process
		Process().create_new_process(g.user.id, 'eigenfaces')
		Process().set_code('recognition')
		# Parse and validate input values
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)
		# Check validation errors
		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(422, "Validation error"), 422

		Eigenfaces.recognize_face()

		# Check recognition errors
		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(400, "Bad request"), 400

		return ResponseHelper.create_response(message), 200

	@app.route('/api/fisher/', methods=['POST'])
	@auth.login_required
	def fisherfaces():
		Utils.reset_singletons()
		# Create new process
		Process().create_new_process(g.user.id, 'fisherfaces')
		Process().set_code('recognition')
		# Parse and validate input values
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings'}
		inputs.set_attributes(request)
		# Check validation errors
		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(422, "Validation error"), 422

		Fisherfaces.recognize_face()

		# Check recognition errors
		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(400, "Bad request"), 400

		return ResponseHelper.create_response(message), 200

	# Generate System Stats Routers
	@app.route('/api/stats/', methods=['POST'])
	def stats():
		Utils.reset_singletons()
		# Parse and validate input values
		inputs = InputParser()
		inputs.validate_attributes = {'extraction_settings', 'recognition_settings', 'stats'}
		inputs.set_attributes(request)

		Process().is_new = False

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(400, "Validate error"), 400

		Stats.statistics()

		return ResponseHelper.create_response(200, message), 200

	# Authorization Routers
	@app.route('/api/token/', methods=['GET'])
	@auth.login_required
	def get_auth_token():
		Utils.reset_singletons()
		# Generate the token
		token = g.user.generate_auth_token()
		return flask.jsonify({'token': token.decode('ascii')})

	@auth.verify_password
	def verify_password(username_or_token, password):
		# First try to authenticate by token
		user = User.verify_auth_token(username_or_token)
		if not user:
			# Try to authenticate with username/password
			user = User.query.filter_by(username=username_or_token).first()
			if not user or not user.verify_password(password):
				abort(401)
				return False

		g.user = user
		return True

	# User Routers
	@app.route('/api/users/')
	@auth.login_required
	def listing_user():
		Utils.reset_singletons()

		if not g.user.is_admin:
			return jsonify({
				'code': 405,
				'message': "List of Users is not allowed",
			}), 405

		result = Users.listing()
		return jsonify(result), 200

	@app.route('/api/users/face/', methods=['POST'])
	@auth.login_required
	def save_image_for_user():
		Utils.reset_singletons()
		inputs = InputParser()
		inputs.set_attributes(request)

		if not ErrorParser().is_empty():
			return ResponseHelper.create_response(422, "Validation error"), 422

		url = Users.save_face_image()

		if not ErrorParser().is_empty():
			db.session.rollback()
			return ResponseHelper.create_response(400, "Bad request"), 400

		return jsonify({
			'image_face': url
		}), 201

	@app.route("/api/users/<user_id>/", methods=['PUT'])
	@auth.login_required
	def update_user(user_id):
		Utils.reset_singletons()

		if int(g.user.id) != int(user_id):
			if not g.user.is_admin:
				return jsonify({
					'code': 405,
					'message': "User is not allowed",
				}), 405

		result = Users.update(user_id)

		if result:
			return jsonify(result), 200
		else:
			return jsonify({
				'code': 404,
				'message': "User not found",
			}), 404

	@app.route("/api/users/me/", methods=['GET'])
	@auth.login_required
	def get_user():
		Utils.reset_singletons()

		# Get user detail
		result = Users.me()

		if result:
			return jsonify(result), 200
		else:
			return jsonify({
				'code': 404,
				'message': "User not found",
			}), 404

	@app.route("/api/users/<user_id>/", methods=['GET'])
	@auth.login_required
	def get_user_by_id(user_id):
		Utils.reset_singletons()

		if not g.user.is_admin:
			return jsonify({
				'code': 405,
				'message': "User is not allowed",
			}), 405

		result = Users.get(user_id)

		if result:
			return jsonify(result), 200
		else:
			return jsonify({
				'code': 404,
				'message': "User not found",
			}), 404


	@app.route("/api/users/logs/", methods=['GET'])
	@auth.login_required
	def get_logs():
		Utils.reset_singletons()
		# Get user processes
		result = Users.logs()
		print("Logs result")
		print(result)
		return jsonify(result), 200

	@app.route("/api/users/logs/<log_id>/", methods=['GET'])
	@auth.login_required
	def get_log_details(log_id):
		Utils.reset_singletons()
		# Get user processes
		result = Users.log_details(log_id)

		if result:
			return jsonify(result), 200
		else:
			return jsonify({
				'code': 404,
				'message': "Process not found",
			}), 404

	@app.route('/api/users/', methods=['POST'])
	def new_user():
		Utils.reset_singletons()
		result = Users.registration()
		return jsonify(result), 201

	# Image routers
	@app.route("/api/images/<image_id>/", methods=['GET'])
	def get_image(image_id):
		Utils.reset_singletons()
		image = Image.get_by_id(image_id)
		if image is None:
			abort(404)
		# Set up image content type headers to show image with URL
		response = make_response(image.image)
		response.headers['Content-Type'] = 'image/jpeg'
		return response

	@app.route("/api/images/<image_id>/", methods=['DELETE'])
	@auth.login_required
	def delete_image(image_id):
		Utils.reset_singletons()

		image = Image.get_by_id(image_id)
		# Check image user permissions
		if not g.user.is_admin:
			if g.user.id != image.user_id:
				return jsonify({
					'code': 405,
					'message': "User has n permission",
				}), 405

		if image is None:
			return jsonify({
				'code': 404,
				'message': "Image not found",
			}), 404

		parent_id = image.parent_id
		# Delete histograms
		Histogram.remove_by_image(image_id)
		Histogram.remove_by_image(parent_id)
		# Delete image
		Image.remove(image_id)
		# Delete image by parent_id
		Image.remove_by_parent(parent_id)
		# Delete image by image id in parent column
		Image.remove_by_parent(image_id)

		return "", 204

	# Testing Routers
	# @app.route("/api/hidden/image/save/")
	# def hidden():
	# 	ID = 23
	# 	import glob
	# 	for filename in glob.glob('static/' + str(ID) + '/*.jpg'):
	# 		base64 = ImageHelper.encode_base64_from_path(filename)
	#
	# 		face, parent_id = ImageHelper.prepare_face_new(base64.decode("utf-8"), 'full')
	# 		image_id = ImageHelper.save_image(face, 'face', ID, parent_id)
	#
	# 	return "Done"

	@app.route("/api/")
	def documentation():
		print("Documentation")
		return send_from_directory(os.path.join('.', '_docs'), 'Dipl.html')

	@app.route("/")
	def hello():
		ErrorParser().add_error('histogram', 'generals.histogram.required')
		return ResponseHelper.create_response(400), 400

	@app.errorhandler(400)
	def bad_request(error):
		print(error)
		return ResponseHelper().create_simple_response(400, getattr(error, 'description', "Bad request")), 400

	@app.errorhandler(401)
	def unauthorized(error):
		print(error)
		return ResponseHelper().create_simple_response(401, "Unauthorized Access"), 401

	@app.errorhandler(422)
	def unprocessed_entity(error):
		return ResponseHelper().create_simple_response(422, "Validation error"), 422

	@app.errorhandler(500)
	def internal_error(error):
		return ResponseHelper().create_simple_response(500, getattr(error, 'description', "Internal server error")), 500

	return app
