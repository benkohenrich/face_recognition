import numpy
import time
from flask import json, g
from sklearn.svm import SVC

from helpers.processhelper import Process as ProcessHelper
from helpers.recognizerhelper import RecognizeHelper
from models.image import Image
from models.user import User

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from helpers.parsers import InputParser, ErrorParser, ResponseParser
from helpers.eigenfaceshelper import EigenfacesHelper

from scipy.spatial import distance as dist
from sklearn.grid_search import GridSearchCV


class EigenfacesRecognizer:
	SCIPY_METHODS = {
		"euclidean": dist.euclidean,
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, recognize_face, number_components=24, method='randomized', compared_face_id=None, whiten = False):
		self.method = method
		self.number_components = number_components
		self.input_parser = InputParser()
		self.compare_face = recognize_face
		self.algorithm = "none"
		self.compared_face_id = compared_face_id
		if whiten is None:
			self.whiten = True
		else:
			if whiten == 'true' or whiten == 1:
				self.whiten = True
			elif whiten == 'false' or whiten == 0:
				self.whiten = False
			else:
				self.whiten = True

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')

		self.algorithm = argument

		switcher = {
			'svm': self.svm_recognize,
			'euclidean': self.scipy_recognize_method,
			"manhattan": self.scipy_recognize_method,
			"chebysev": self.scipy_recognize_method,
			"cosine": self.scipy_recognize_method,
			"braycurtis": self.scipy_recognize_method
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()

	def svm_recognize(self):

		start_time = time.time()
		model, X_pca, y, y_images, total_image = EigenfacesHelper.prepare_data(self.number_components, self.method, self.whiten)

		# Prepare Image to recognize
		test = EigenfacesHelper.prepare_image(self.compare_face, 'test')
		test_pca = model.transform(test)

		# Train a SVM classification model
		print("Fitting the classifier to the training set")
		param_grid = {
			'C': [1, 5, 10, 50, 100],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
		}

		# Normalize data
		X_train, X_test = RecognizeHelper.normalize_data(X_pca, test_pca)

		clf = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, n_jobs=10)
		clf = clf.fit(X_train, y)

		Y_pred = clf.predict(X_test)
		percentage_array = clf.predict_proba(X_test)

		predict_user_id = int(Y_pred[0])
		predict_user = User.query.filter(User.id == predict_user_id).first()
		percentage = numpy.sum(percentage_array)

		print(type(percentage), percentage)

		if self.number_components == 0:
			self.number_components = "auto"

		process = {
			"parameters": {
				'number_components': self.number_components,
				'method': self.method,
				'whiten': self.whiten,
				"algorithm": self.algorithm,
				"recognize_eigenfaces": json.dumps(X_test[0].tolist()),
				"total_compared_eigenfaces": total_image,
				'similarity_percentage': percentage * 100,
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': time.time() - start_time
			}
		}

		ResponseParser().add_process('recognition', process)

	def scipy_recognize_method(self):

		start_time = time.time()

		if self.SCIPY_METHODS[self.algorithm] is None:
			ErrorParser().add_error('algorithm', '')
			return
		else:
			method = self.SCIPY_METHODS[self.algorithm]

		model, X_pca, y, images, total_image = EigenfacesHelper.prepare_data(self.number_components, self.method)

		test = EigenfacesHelper.prepare_image(self.compare_face, 'test')
		test = model.transform(test)

		distances = []
		distance = None
		# run through test images (usually one)
		X_train, X_test = RecognizeHelper.normalize_data(X_pca, test)

		for j, ref_pca in enumerate(X_train):
			dist = method(ref_pca, X_test[0])
			distances.append((dist, y[j], images[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		found_image_ID = min(distances)[2]

		percentage = RecognizeHelper.calculate_percentage_for_distance_metric_methods(g.user.id, distance, distances)
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		predict_user_id = int(found_ID)
		predict_user = User.query.filter(User.id == found_ID).first()
		process = {
			"parameters": {
				'number_components': self.number_components,
				'method': self.method,
				"algorithm": self.algorithm,
				'whiten': self.whiten,
				"recognize_eigenfaces": json.dumps(test[0].tolist()),
				"total_compared_eigenfaces": total_image,
				'similarity_percentage': percentage,
				'distance': str(distance),
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': time.time() - start_time,
			}
		}

		ResponseParser().add_process('recognition', process)
		ResponseParser().add_image('recognition', 'predict_image', found_image_ID)
		ResponseParser().add_image('recognition', 'compared_image', ProcessHelper().face_image_id)

