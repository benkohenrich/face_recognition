# from matplotlib.mlab import PCA
from PIL import Image
from flask import json
from scipy import ndimage, misc
from sklearn.svm import SVC

from helpers.imagehelper import ImageHelper

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from sklearn.decomposition import PCA, pca
import numpy as np
import glob
import cv2
import math
import os.path
import string
from helpers.parsers import InputParser, ErrorParser, ResponseParser
from helpers.eigenfaceshelper import EigenfacesHelper
from models.image import Image as ImageModel
from scipy.spatial import distance as dist
from sklearn.grid_search import GridSearchCV


class FisherfacesRecognizer:

	SCIPY_METHODS = {
		"euclidian" : dist.euclidean,
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, recognize_face, number_components=100, method='randomized'):
		self.method = method
		self.number_components = number_components
		self.input_parser = InputParser()
		self.compare_face = recognize_face
		self.algorithm = "none"
		self.algorithm = "none"

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')
		self.algorithm = argument
		switcher = {
			# 'svm': self.svm_recognize,
			'euclidian': self.scipy_recognize_method,
			"manhattan":self.scipy_recognize_method,
			"chebysev":self.scipy_recognize_method,
			"cosine":self.scipy_recognize_method,
			"braycurtis":self.scipy_recognize_method
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()

	def scipy_recognize_method(self):

		if self.SCIPY_METHODS[self.algorithm] is None:
			ErrorParser().add_error('algorithm', '')
			return
		else:
			method = self.SCIPY_METHODS[self.algorithm]

		model, X_pca, y, images, total_image = EigenfacesHelper.prepare_data(self.number_components, self.method)

		test = EigenfacesHelper.prepare_image(self.comparing_face, 'test')
		test = model.transform(test)

		distances = []
		distance = None
		# run through test images (usually one)

		X_train, X_test = RecognizeHelper.normalize_data(X_pca, test)

		for j, ref_pca in enumerate(X_train):
			print(len(ref_pca), " = ", len(X_test[0]))
			dist = method(ref_pca, X_test[0])
			print("Scipy Distance: ", float("{0:.50f}".format(dist)), " UserID:", y[j] ," ImageID: ", images[j])
			distances.append((dist, y[j], images[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		found_image_ID = min(distances)[2]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		predict_user_id = int(found_ID)
		predict_user = User.query.filter(User.id == found_ID).first()
		process = {
			"parameters": {
				'n_components': self.number_components,
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
		ResponseParser().add_image('recognition', 'predict_image', found_image_ID)
	# def euclidian_recognize(self):
	#
	# 	model , X_pca, y, total_image = EigenfacesHelper.cross_validate(self.num_eigenfaces, self.method)
	#
	# 	npimg = ImageHelper.decode_base64_to_filename(self.comparing_face)
	#
	#
	# 	# npimg = ImageHelper.convert_base64_image_to_numpy(self.comparing_face)
	#
	# 	img_color = cv2.imdecode(npimg, 1)
	#
	# 	img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
	# 	h, s, v = cv2.split(img_gray)
	# 	v += 255
	# 	final_hsv = cv2.merge((h, s, v))
	# 	# img_gray = cv2.equalizeHist(img_gray)
	# 	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	# 	cv2.imwrite(npimg, img)
	# 	exit()
	# 	# ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
	# 	# X = np.zeros([1, 100 * 100], dtype='int8')
	#
	# 	test = img_gray.flat
	#
	# 	print("After flat: " , test)
	# 	test = model.transform(test)
	#
	# 	distances = []
	# 	# run through test images (usually one)
	# 	for j, ref_pca in enumerate(X_pca):
	# 		print("TEST VECtOR: ", test[0])
	# 		print("TRAIN VECtOR: ", ref_pca)
	#
	# 		dist = math.sqrt(sum([diff ** 2 for diff in (ref_pca - test[0])]))
	# 		print("Distance: ", float("{0:.20f}".format(dist)), " UserID:", y[j])
	# 		distances.append((dist, y[j]))
	#
	# 	found_ID = min(distances)[1]
	# 	distance = min(distances)[0]
	# 	print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")
	#
	# 	process = {
	# 		"parameters": {
	# 			'num_eigenfaces': self.num_eigenfaces,
	# 			'method': self.method,
	# 			"algorithm": self.algorithm,
	# 			"recognize_eigenfaces": json.dumps(test[0].tolist()),
	# 			"total_compared_histograms": total_image,
	# 			'distance': str(distance),
	# 			"predict_user": {
	# 				"id": int(found_ID),
	# 				"name": "",
	# 				"main_image": ""
	# 			},
	# 		},
	# 		"messages": {
	#
	# 		},
	# 		"metadata": {
	#
	# 		}
	# 	}
	#
	# 	ResponseParser().add_process('recognition', process)
