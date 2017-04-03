from flask import json
from sklearn.svm import SVC

from models.image import Image
from models.user import User

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

import cv2
import math

from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser, ErrorParser, ResponseParser
from helpers.eigenfaceshelper import EigenfacesHelper

from scipy.spatial import distance as dist
from sklearn.grid_search import GridSearchCV


class EigenfacesRecognizer:
	SCIPY_METHODS = {
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, EIGENFACE, NUM_EIGENFACES=24, METHOD='randomized'):
		# self.points = int(num_points)
		self.method = METHOD
		self.num_eigenfaces = NUM_EIGENFACES
		self.input_parser = InputParser()
		self.COMPARING_EIGENFACE = EIGENFACE

		self.algorithm = "none"

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')
		self.algorithm = argument
		switcher = {
			'svm': self.svm_recognize,
			'euclidian': self.euclidian_recognize,
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
		print("NO SVM FOR NOW")

		model, X_pca, y, total_image = EigenfacesHelper.cross_validate(self.num_eigenfaces, self.method)

		npimg = ImageHelper.convert_base64_image_to_numpy(self.COMPARING_EIGENFACE)

		img_color = cv2.imdecode(npimg, 1)

		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)

		# ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
		# X = np.zeros([1, 100 * 100], dtype='int8')

		test = img_gray.flat

		test = model.transform(test)

		################################################################################
		# Train a SVM classification model

		print("Fitting the classifier to the training set")
		param_grid = {
			'C': [1, 5, 10, 50, 100],
			'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
		}

		clf = GridSearchCV(SVC(kernel='rbf'), param_grid, n_jobs=1)
		clf = clf.fit(X_pca, y)

		y_pred = clf.predict(test)

		predict_user_id = int(y_pred[0])

		predict_user = User.query.filter(User.id == predict_user_id).first()

		process = {
			"parameters": {
				'num_eigenfaces': self.num_eigenfaces,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_eigenfaces": json.dumps(test[0].tolist()),
				"total_compared_faces": total_image,
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
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

		model, X_pca, y, total_image = EigenfacesHelper.cross_validate(self.num_eigenfaces, self.method)

		npimg = ImageHelper.convert_base64_image_to_numpy(self.COMPARING_EIGENFACE)

		img_color = cv2.imdecode(npimg, 1)

		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)

		# ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
		# X = np.zeros([1, 100 * 100], dtype='int8')

		test = img_gray.flat
		# print(test)
		test = model.transform(test)
		# exit()
		distances = []
		distance = None
		# run through test images (usually one)
		for j, ref_pca in enumerate(X_pca):
			dist = method(ref_pca, test[0])
			print("Scipy Distance: ", float("{0:.20f}".format(dist)), " UserID:", y[j])
			distances.append((dist, y[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		predict_user_id = int(found_ID)
		predict_user = User.query.filter(User.id == found_ID).first()
		process = {
			"parameters": {
				'num_eigenfaces': self.num_eigenfaces,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_eigenfaces": json.dumps(test[0].tolist()),
				"total_compared_histograms": total_image,
				'distance': str(distance),
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)

	def euclidian_recognize(self):

		model, X_pca, y, total_image = EigenfacesHelper.cross_validate(self.num_eigenfaces, self.method)

		npimg = ImageHelper.convert_base64_image_to_numpy(self.COMPARING_EIGENFACE)

		img_color = cv2.imdecode(npimg, 1)

		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)

		# ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
		# X = np.zeros([1, 100 * 100], dtype='int8')

		test = img_gray.flat

		print("After flat: ", test)
		test = model.transform(test)

		distances = []
		# run through test images (usually one)
		for j, ref_pca in enumerate(X_pca):

			dist = math.sqrt(sum([diff ** 2 for diff in (ref_pca - test[0])]))
			# print("Distance: ", float("{0:.20f}".format(dist)), " UserID:", y[j])
			distances.append((dist, y[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		predict_user_id = int(found_ID)
		predict_user = User.query.filter(User.id == found_ID).first()

		process = {
			"parameters": {
				'num_eigenfaces': self.num_eigenfaces,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_eigenfaces": json.dumps(test[0].tolist()),
				"total_compared_histograms": total_image,
				'distance': str(distance),
				"predict_user": {
					"id": predict_user_id,
					"name": predict_user.name,
					"email": predict_user.username,
					"main_image": Image.avatar_path(predict_user.id)
				},
			},
			"metadata": {
				'process_time': '',
				'process_mem_use': ''
			}
		}

		ResponseParser().add_process('recognition', process)
