import base64
from PIL import Image
import cv2
from flask import json

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from helpers.imagehelper import ImageHelper
from helpers.parsers import InputParser
from helpers.parsers import ResponseParser
# from matplotlib import pyplot as plt


class HistogramMaker(object):
	@staticmethod
	def create_histogram_from_image(image_path):

		options = InputParser().extraction_settings

		if type(image_path).__module__ == np.__name__:
			im = cv2.imdecode(image_path, 1)
		else:
			im = cv2.imread(image_path)

		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		try:
			radius = int(options['radius'])
		except KeyError:
			radius = 3

		# Number of points to be considered as neighbourers
		try:
			no_points = int(options['points']) * radius
		except KeyError:
			no_points = 8 * radius

		try:
			method = options['method']
		except KeyError:
			method = 'uniform'

		# Uniform LBP is used
		lbp = local_binary_pattern(im_gray, no_points, radius, method=method)

		# Calculate the histogram
		# print(plt.hist(im.ravel(), 256, [0, 256]))
		# plt.show()
		x = itemfreq(lbp.ravel())

		# Normalize the histogram
		hist = x[:, 1] / sum(x[:, 1])

		result = {'radius': radius, 'points': no_points, 'histogram': hist, 'method': method}
		result_response = {'radius': radius, 'points': no_points, 'histogram':  json.dumps(hist.tolist()), 'method': method}

		process = {
			"parameters" : result_response,
			"messages" : {

			},
			"images" : {

			},
			"metadata" : {

			}
		}

		ResponseParser().add_process('extraction', process)

		return result

	@staticmethod
	def create_histogram_from_b64(base64_string):

		# if isinstance(base64_string, (bytes, bytearray)):

		base64_string = ImageHelper.encode_base64(base64_string)

		# if not isinstance(base64_string, str):

		base64_string = str(base64_string, 'utf-8')

		decoded = base64.b64decode(base64_string)

		npimg = np.fromstring(decoded, dtype=np.uint8)

		# print(type(npimg))
		#
		return HistogramMaker.create_histogram_from_image(npimg)