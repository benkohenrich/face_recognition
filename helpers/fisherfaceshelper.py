import cv2
from flask import current_app
import numpy as np

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from helpers.parsers import InputParser
from helpers.parsers import ResponseParser
from models.image import Image as ImageModel
from sklearn import lda


class FisherfaceHelper(object):
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
			X[c, :] = FisherfaceHelper.prepare_image(image.image)
			y.append(image.user_id)
			c += 1

		lda_model = lda.LDA(n_components=num_eigenfaces).fit(X, y)
		X_lda = lda_model.transform(X)

		return lda_model, X_lda, y, total_image
