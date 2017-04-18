import base64
import io

import matplotlib
from PIL import Image
import cv2
from flask import g
from flask import json

from helpers.processhelper import Process

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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class HistogramMaker(object):


	@staticmethod
	def create_histogram_from_image(image):

		# GET SETTINGS
		options = InputParser().extraction_settings

		# OPEN IMAGE
		if type(image).__module__ == np.__name__:
			im = cv2.imdecode(image, 1)
		else:
			im = cv2.imread(image)

		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		# GRAYSCALE IMAGE
		gray_image = cv2.imencode('.jpg', im_gray)[1].tostring()

		if Process().is_new:
			face_grey_id = ImageHelper.save_image(gray_image, 'face_grey', g.user.id)

		equ = cv2.equalizeHist(im_gray)

		if Process().is_new:
			face_equalized_id = ImageHelper.save_numpy_image(equ, 'face_equalized', g.user.id)

		plt.hist(equ.ravel(), 256, [0, 256])

		if Process().is_new:
			histogram_graph_id = ImageHelper.save_plot_image(plt, 'histogram_graph', g.user.id)

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

		lbp = local_binary_pattern(im_gray, no_points, radius, method=method)

		x = itemfreq(lbp.ravel())

		# Normalize the histogram
		hist = x[:, 1] / sum(x[:, 1])

		result = {
			'radius': radius,
			'points': no_points,
			'histogram': hist,
			'method': method,
		}

		result_response = {
			'radius': radius,
			'points': no_points,
			'histogram': json.dumps(hist.tolist()),
			'method': method
		}

		process = {
			"parameters": result_response,
			"messages": {},
			"metadata": {}
		}

		ResponseParser().add_process('extraction', process)

		if Process().is_new:
			ResponseParser().add_image('extraction', 'face_grey', face_grey_id)
			ResponseParser().add_image('extraction', 'face_equalized', face_equalized_id)
			ResponseParser().add_image('extraction', 'histogram_graph', histogram_graph_id)

		return result

	@staticmethod
	def create_histogram_from_b64(base64_string):

		return HistogramMaker.create_histogram_from_image(ImageHelper.convert_base64_to_numpy(base64_string))

	@staticmethod
	def np_hist_to_cv(np_histogram_output):
		counts = np_histogram_output
		return_value = counts.ravel().astype('float32')
		return return_value
