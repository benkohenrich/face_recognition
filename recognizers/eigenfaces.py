# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
import numpy as np
import glob
import cv2
import math
import os.path
import string
from helpers.parsers import InputParser
from models.image import Image


class EigenfacesRecognizer:

	IMG_RES = 100 * 120  # img resolution

	def __init__(self, EIGENFACE, NUM_EIGENFACES=24, METHOD='randomized'):
		# self.points = int(num_points)
		self.METHOD = METHOD
		self.NUM_EIGENFACES = NUM_EIGENFACES
		self.input_parser = InputParser()
		self.COMPARING_EIGENFACE = EIGENFACE

	@staticmethod
	def prepare_image(filename):
		img_color = cv2.imread(filename)
		img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.equalizeHist(img_gray)
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
		all_image = Image.query.all()
		total_image = Image.query.count()

		# Create an array with flattened images X
		# and an array with ID of the people on each image y
		X = np.zeros([total_image, self.IMG_RES], dtype='int8')
		y = []

		# Populate training array with flattened imags from subfolders of train_faces and names
		c = 0
		for image in all_image:
			# print(c)
			X[c, :] = self.prepare_image(image.image)
			y.append(image.user_id)
			c += 1

		pca = PCA(n_components=self.NUM_EIGENFACES, whiten=True, svd_solver=self.METHOD).fit(X)
		# print(X.shape)
		X_pca = pca.transform(X)

		# X = np.zeros([len(test_faces), self.IMG_RES], dtype='int8')
		# for i, face in enumerate(test_faces):
		# 	X[i, :] = self.prepare_image(face)
		distances = []
		# run through test images (usually one)
		for j, ref_pca in enumerate(X_pca):

			dist = math.sqrt(sum([diff ** 2 for diff in (ref_pca - self.COMPARING_EIGENFACE)]))
			distances.append((dist, y[j]))

		found_ID = min(distances)[1]
		print("Identified (result: " + str(found_ID) + " - dist - " + str(min(distances)[0]) + ")")