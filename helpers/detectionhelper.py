import os
import base64
import PIL
import cv2
import numpy as np

from flask import current_app
from PIL import Image
from models.image import Image as ModelImage

class DetectionHelper(object):

	@staticmethod
	def haar_cascade_detect(filepath):

		face_cascade = cv2.CascadeClassifier(
			'/Users/henrichbenko/School/OpenCV/opencv-3.1.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')

		img = cv2.imread(filepath)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			crop_img = img[y:(y+h), x:(x+w)]
			cv2.imwrite(filepath, crop_img)

		return filepath
