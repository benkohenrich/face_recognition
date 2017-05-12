# from matplotlib.mlab import PCA
import numpy
from PIL import Image
from flask import json, g
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from helpers.recognizerhelper import RecognizeHelper
from models.user import User
from models.image import Image as ImageModel

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from helpers.parsers import InputParser, ErrorParser, ResponseParser
from helpers.processhelper import Process as ProcessHelper
from helpers.eigenfaceshelper import EigenfacesHelper
from scipy.spatial import distance as dist


class FisherfacesRecognizer:

	SCIPY_METHODS = {
		"euclidean" : dist.euclidean,
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, recognize_face, number_components=100, tolerance=0.0001):
		self.tolerance = tolerance
		self.number_components = number_components
		self.input_parser = InputParser()
		self.compare_face = recognize_face
		self.algorithm = "none"

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')
		self.algorithm = argument
		switcher = {
			'svm': self.svm_recognize,
			'euclidean': self.scipy_recognize_method,
			"manhattan":self.scipy_recognize_method,
			"chebysev":self.scipy_recognize_method,
			"cosine":self.scipy_recognize_method,
			"braycurtis":self.scipy_recognize_method
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()

	def svm_recognize(self):
		model, X_pca, y, y_images, total_image = EigenfacesHelper.prepare_data_fisher(self.number_components, self.tolerance)

		# Prepare Image to recognize
		test = EigenfacesHelper.prepare_image(self.compare_face, 'test')
		test_pca = model.transform(test)

		################################################################################
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
		percentage_array = clf.predict_proba(X_test)
		Y_pred = clf.predict(X_test)
		predict_user_id = int(Y_pred[0])
		predict_user = User.query.filter(User.id == predict_user_id).first()

		percentage = numpy.sum(percentage_array)

		process = {
			"parameters": {
				'n_components': self.number_components,
				'tolerance': self.tolerance,
				"algorithm": self.algorithm,
				"recognize_fisherfaces": json.dumps(X_test[0].tolist()),
				"total_compared_faces": total_image,
				'similarity_percentage': percentage * 100,
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": ImageModel.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)

	def scipy_recognize_method(self):

		if self.SCIPY_METHODS[self.algorithm] is None:
			ErrorParser().add_error('algorithm', '')
			return
		else:
			method = self.SCIPY_METHODS[self.algorithm]

		model, X_pca, y, images, total_image = EigenfacesHelper.prepare_data_fisher(self.number_components, self.tolerance)

		test = EigenfacesHelper.prepare_image(self.compare_face, 'test')
		test = model.transform(test)

		distances = []
		distance = None

		X_train, X_test = RecognizeHelper.normalize_data(X_pca, test)

		for j, ref_pca in enumerate(X_train):
			dist = method(ref_pca, X_test[0])
			distances.append((dist, y[j], images[j]))

		distance = min(distances)[0]
		found_ID = min(distances)[1]
		found_image_ID = min(distances)[2]
		percentage = RecognizeHelper.calculate_percentage_for_distance_metric_methods(g.user.id, distance, distances)
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		predict_user_id = int(found_ID)
		predict_user = User.query.filter(User.id == found_ID).first()

		process = {
			"parameters": {
				'n_components': self.number_components,
				'tolerance': self.tolerance,
				"algorithm": self.algorithm,
				"recognize_fisherfaces": json.dumps(test[0].tolist()),
				'similarity_percentage': percentage,
				"total_compared_histograms": total_image,
				'distance': str(distance),
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": ImageModel.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)
		ResponseParser().add_image('recognition', 'predict_image', found_image_ID)
		ResponseParser().add_image('recognition', 'compared_image', ProcessHelper().face_image_id)
