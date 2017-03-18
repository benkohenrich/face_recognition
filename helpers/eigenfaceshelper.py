import base64

import LDA as LDA
from PIL import Image
import cv2
# from dask.tests.test_base import np
from flask import current_app
from flask import g
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
from models.image import Image as ImageModel


class EigenfacesHelper(object):
	@staticmethod
	def prepare_image(file_bytes):
		nparr = np.fromstring(file_bytes, np.uint8)
		img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)

		return img_gray.flat

	@staticmethod
	def cross_validate(num_eigenfaces, method):
		all_image = ImageModel.get_all_to_extraction()
		total_image = len(all_image)

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = np.zeros([total_image, current_app.config['IMG_RES']], dtype='int8')
		y = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in all_image:
			X[c, :] = EigenfacesHelper.prepare_image(image.image)
			y.append(image.user_id)
			c += 1

		pca = PCA(n_components=num_eigenfaces, whiten=True, svd_solver=method).fit(X)
		X_pca = pca.transform(X)

		return pca, X_pca, y, total_image

	@staticmethod
	def create_base64_to_eigenface(base64face):
		options = InputParser()

	#
	# num_eigenfaces = options.__getattr__('number_eigenfaces')
	# svd_solver = options.__getattr__('method')
	#
	# # print(base64face)
	# npimg = ImageHelper.convert_base64_image_to_numpy(base64face)
	#
	# img_color = cv2.imdecode(npimg, 1)
	#
	# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
	# img_gray = cv2.equalizeHist(img_gray)
	#
	# # ImageHelper.save_numpy_image(img_gray, 'test', g.user.id)
	# X = np.zeros([5, 100*100], dtype='int8')
	# #
	# X[0, :] = img_gray.flat
	# X[1, :] = img_gray.flat
	# X[2, :] = img_gray.flat
	# X[3, :] = img_gray.flat
	# X[4, :] = img_gray.flat
	#
	# # print(X)
	# #
	# # print(X)
	# pca = PCA(n_components=24, whiten=True, svd_solver='randomized').fit(X)
	# # print(pca)
	# # X_pca = pca.transform(X)
	# print("################")
	# print(pca)

	# print(base64face)
	# options = InputParser()
	#
	# num_eigenfaces = options.__getattr__('number_eigenfaces')
	# svd_solver = options.__getattr__('method')
	#
	# #
	#
	# with open('tmp.png', 'wb') as f:
	# 	f.write(base64face)
	#
	# img_color = cv2.imread('tmp.png')
	# # img_color = cv2.imdecode('tmp.png', 1)
	#
	# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
	# img_gray = cv2.equalizeHist(img_gray)
	#
	# X = img_gray.flat
	#
	# print("********EXTRACTION FLAT**********")
	# print(X)
	# print("*********************************")
	# pca = PCA(n_components=int(num_eigenfaces), whiten=True, svd_solver=svd_solver).fit(X)
	# # exit()
	# X_pca = pca.transform(X)
	#
	# print(type(X_pca))
	# print(X_pca)
	# print("###############################")
	# result_response = {
	# 	'number_eigenfaces': num_eigenfaces,
	# 	'method': svd_solver,
	# 	'X_pca': json.dumps(X_pca.tolist()),
	# }
	#
	# process = {
	# 	"parameters": result_response,
	# 	"messages": {
	#
	# 	},
	# 	"images": {
	#
	# 	},
	# 	"metadata": {
	#
	# 	}
	# }
	#
	# ResponseParser().add_process('extraction', process)
	#
	# return X_pca
