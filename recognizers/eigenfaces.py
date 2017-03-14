# from matplotlib.mlab import PCA
from PIL import Image
from scipy import ndimage, misc

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

from sklearn.decomposition import PCA
import numpy as np
import glob
import cv2
import math
import os.path
import string
from helpers.parsers import InputParser
from models.image import Image as ImageModel


class EigenfacesRecognizer:
	IMG_RES = 100 * 100  # img resolution

	def __init__(self, EIGENFACE, NUM_EIGENFACES=24, METHOD='randomized'):
		# self.points = int(num_points)
		self.METHOD = METHOD
		self.NUM_EIGENFACES = NUM_EIGENFACES
		self.input_parser = InputParser()
		self.COMPARING_EIGENFACE = EIGENFACE

	@staticmethod
	def prepare_image(file_bytes):

		# 	sbuf = StringIO()
		# 	sbuf.write(str(filename))
		# 	img_color = Image.open(sbuf)
		# 	print(file_bytes)
		# 	print(type(file_bytes))

		nparr = np.fromstring(file_bytes, np.uint8)

		img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		# img_color = cv2.imread(file_bytes)
		# img_color = misc.imresize(img_color, (100, 120))
		# img_color.append(image_resized)
		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)
		print(img_gray.flat)

		return img_gray.flat

	def recognize(self):
		argument = self.input_parser.__getattr__('algorithm')

		switcher = {
			'svm': self.svm_recognize,
			'euclidian': self.euclidian_recognize
		}

		# Get the function from switcher dictionary
		func = switcher.get(argument, lambda: "nothing")

		# Execute the function
		func()

	def svm_recognize(self):
		print("NO SVM FOR NOW")

	def euclidian_recognize(self):
		all_image = ImageModel.get_all_to_extraction()
		total_image = ImageModel.query.count()

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = np.zeros([total_image, self.IMG_RES], dtype='int8')
		# X = []
		y = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in all_image:
			# print(c)
			# X.append(self.prepare_image(image.image))
			X[c, :] = self.prepare_image(image.image)
			y.append(image.user_id)
			c += 1

		print(type(X))
		# print(type(X2))
		# X = np.array(X)
		pca = PCA(n_components=self.NUM_EIGENFACES, whiten=True, svd_solver=self.METHOD).fit(X)
		# print(X.shape)
		X_pca = pca.transform(X)

		print(X_pca)
		# X = np.zeros([len(test_faces), self.IMG_RES], dtype='int8')
		# for i, face in enumerate(test_faces):
		# 	X[i, :] = self.prepare_image(face)
		distances = []
		# run through test images (usually one)
		for j, ref_pca in enumerate(X_pca):
			print(ref_pca)
			print(self.COMPARING_EIGENFACE)
			dist = math.sqrt(sum([diff ** 2 for diff in (ref_pca - self.COMPARING_EIGENFACE)]))
			distances.append((dist, y[j]))

		found_ID = min(distances)[1]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(min(distances)[0]) + ")")
