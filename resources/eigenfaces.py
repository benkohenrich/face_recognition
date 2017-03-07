import os

import cv2
from flask_restful import Resource

from sklearn.decomposition import RandomizedPCA
import numpy as np
import glob
import cv2
import math
import os.path
import string

from gzip import GzipFile

import numpy as np
import pylab as pl

# from scikits.learn.grid_search import GridSearchCV
# from scikits.learn.metrics import classification_report
# from scikits.learn.metrics import confusion_matrix
# from scikits.learn.pca import RandomizedPCA
# from scikits.learn.svm import SVC

class Eigenfaces(Resource):
	@staticmethod
	def test():
		model = cv2.face.createEigenFaceRecognizer()
		name = model.name

		print(name)


