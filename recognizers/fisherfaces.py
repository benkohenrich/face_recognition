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
		"manhattan": dist.cityblock,
		"chebysev": dist.chebyshev,
		"cosine": dist.cosine,
		"braycurtis": dist.braycurtis,
	}

	def __init__(self, face, NUM_EIGENFACES=24, METHOD='randomized'):
		# self.points = int(num_points)
		self.method = METHOD
		self.num_eigenfaces = NUM_EIGENFACES
		self.input_parser = InputParser()
		self.comparing_face = face

		self.algorithm = "none"

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')
		self.algorithm = argument
		switcher = {
			# 'svm': self.svm_recognize,
			'euclidian': self.euclidian_recognize,
			# "manhattan":self.scipy_recognize_method,
			# "chebysev":self.scipy_recognize_method,
			# "cosine":self.scipy_recognize_method,
			# "braycurtis":self.scipy_recognize_method
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()


	def euclidian_recognize(self):

		model , X_pca, y, total_image = EigenfacesHelper.cross_validate(self.num_eigenfaces, self.method)

		npimg = ImageHelper.convert_base64_image_to_numpy(self.comparing_face)

		img_color = cv2.imdecode(npimg, 1)

		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)

		# ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
		# X = np.zeros([1, 100 * 100], dtype='int8')

		test = img_gray.flat

		print("After flat: " , test)
		test = model.transform(test)

		distances = []
		# run through test images (usually one)
		for j, ref_pca in enumerate(X_pca):
			print("TEST VECtOR: ", test[0])
			print("TRAIN VECtOR: ", ref_pca)

			dist = math.sqrt(sum([diff ** 2 for diff in (ref_pca - test[0])]))
			print("Distance: ", float("{0:.20f}".format(dist)), " UserID:", y[j])
			distances.append((dist, y[j]))

		found_ID = min(distances)[1]
		distance = min(distances)[0]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(distance) + ")")

		process = {
			"parameters": {
				'num_eigenfaces': self.num_eigenfaces,
				'method': self.method,
				"algorithm": self.algorithm,
				"recognize_eigenfaces": json.dumps(test[0].tolist()),
				"total_compared_histograms": total_image,
				'distance': str(distance),
				"predict_user": {
					"id": int(found_ID),
					"name": "",
					"main_image": ""
				},
			},
			"messages": {

			},
			"metadata": {

			}
		}

		ResponseParser().add_process('recognition', process)
