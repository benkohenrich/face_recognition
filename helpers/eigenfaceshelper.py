import cv2
from flask import current_app
from sklearn.decomposition import PCA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler

from helpers.imagehelper import ImageHelper
from helpers.processhelper import Process as ProcessHelper

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from models.image import Image as ImageModel


class EigenfacesHelper(object):

	@staticmethod
	def prepare_image(file_bytes, type='train'):
		if type == 'test':
			npimg = ImageHelper.convert_base64_image_to_numpy(file_bytes)
		else:
			npimg = np.fromstring(file_bytes, np.uint8)

		img_color = cv2.imdecode(npimg, 1)
		adjusted = ImageHelper.adjust_gamma(img_color, gamma=1.5)
		img_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)
		return img_gray.flat

	@staticmethod
	def prepare_data(n_components=100, method="randomized"):

		all_image = ImageModel.get_all_to_extraction(ProcessHelper.face_image_id)
		total_image = len(all_image)

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = np.zeros([total_image, current_app.config['IMG_RES']], dtype='int8')
		y = []
		images = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in all_image:
			X[c, :] = EigenfacesHelper.prepare_image(image.image)
			y.append(image.user_id)
			images.append(image.id)
			c += 1

		# Train data with PCA
		pca = PCA(n_components=n_components, whiten=True, svd_solver=method)
		X_pca = pca.fit_transform(X)

		return pca, X_pca, y, images, total_image

	@staticmethod
	def prepare_data_fisher(n_components=100, method="svd"):

		all_image = ImageModel.get_all_to_extraction()
		total_image = len(all_image)

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = np.zeros([total_image, current_app.config['IMG_RES']], dtype='int8')
		y = []
		images = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in all_image:
			X[c, :] = EigenfacesHelper.prepare_image(image.image)
			y.append(image.user_id)
			images.append(image.id)
			c += 1

		# Train data with LDA
		lda = LinearDiscriminantAnalysis(n_components=n_components, solver=method)
		X_pca = lda.fit_transform(X)

		return lda, X_pca, y, images, total_image

