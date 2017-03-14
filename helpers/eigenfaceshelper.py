import base64
from PIL import Image
import cv2
# from dask.tests.test_base import np
from flask import json
from sklearn.decomposition import PCA
import numpy as np
from helpers.imagehelper import ImageHelper

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from helpers.parsers import InputParser
from helpers.parsers import ResponseParser

class EigenfacesHelper(object):

	@staticmethod
	def create_base64_to_eigenface(base64face):

		options = InputParser()

		num_eigenfaces = options.__getattr__('number_eigenfaces')
		svd_solver = options.__getattr__('method')

		npimg = ImageHelper.convert_base64_to_numpy(base64face)

		img_color = cv2.imdecode(npimg, 1)

		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)
		X = img_gray.flat

		print("********EXTRACTION FLAT**********")
		print(X)
		print("*********************************")
		pca = PCA(n_components=int(num_eigenfaces), whiten=True, svd_solver=svd_solver).fit(X)

		X_pca = pca.transform(X)

		print(type(X_pca))
		result_response = {
			'number_eigenfaces': num_eigenfaces,
			'method': svd_solver,
			'X_pca': json.dumps(X_pca.tolist()),
		}

		process = {
			"parameters": result_response,
			"messages": {

			},
			"images": {

			},
			"metadata": {

			}
		}

		ResponseParser().add_process('extraction', process)

		return X_pca
