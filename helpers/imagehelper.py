import os
import base64
import PIL
import binascii
import cv2
import numpy as np
import time

from flask import current_app, g, abort
from PIL import Image

from helpers.detectionhelper import DetectionHelper
from helpers.parsers import ResponseParser, InputParser, ErrorParser
from helpers.processhelper import Process

from models.image import Image as ModelImage


class ImageHelper(object):
	crop_string = {
		'data:image/png;base64,',
		'data:image/jpeg;base64,',
		'data:image/jpg;base64,'
	}

	@staticmethod
	def decode_base64_to_filename(img_string, filename="tmp.jpg"):

		for crop in ImageHelper.crop_string:
			img_string = img_string.replace(crop, "")

		imgdata = base64.b64decode(img_string)
	
		path = current_app.config['TEMP_PATH'] + str(time.time()) + filename

		with open(path, 'wb') as f:
			f.write(imgdata)
			f.close()

		return path

	@staticmethod
	def image_bytes_to_filename(image_bytes, filename="tmp.jpg"):
		path = current_app.config['TEMP_PATH'] + str(time.time()) + filename

		with open(path, 'wb') as f:
			f.write(image_bytes)
			f.close()

		return path

	@staticmethod
	def decode_base64(img_string):
		for crop in ImageHelper.crop_string:
			img_string = img_string.replace(crop, "")

		return base64.b64decode(img_string)

	@staticmethod
	def crop_type_base64(base64_string):
		for crop in ImageHelper.crop_string:
			base64_string = base64_string.replace(crop, "")

		return base64_string

	@staticmethod
	def encode_base64_from_path(image_path):
		encoded_string = ""

		with open(image_path, "rb") as image_file:
			encoded_string = base64.b64encode(image_file.read())

		return encoded_string

	@staticmethod
	def encode_base64(image_path):
		return base64.b64encode(image_path)

	@staticmethod
	def minimalize(image_path, basewidth=0):

		if basewidth == 0:
			basewidth = current_app.config.get('BASE_WIDTH')

		img = Image.open(image_path)

		width_percent = (basewidth / float(img.size[0]))

		height_size = int((float(img.size[1]) * float(width_percent)))

		img = img.resize((basewidth, height_size), PIL.Image.ANTIALIAS)

		img.save(image_path)

	def minimalize_face(image_path):

		basewidth = current_app.config.get('BASE_WIDTH')

		img = Image.open(image_path)

		height_size = current_app.config.get('BASE_WIDTH')
		img = img.resize((basewidth, height_size), PIL.Image.ANTIALIAS)

		img.save(image_path)

	@staticmethod
	def delete_image(path):
		os.unlink(path)

	@staticmethod
	def prepare_face(face, face_type='face'):

		full_id = None

		image_path = ImageHelper.decode_base64_to_filename(face)
		image_path_big = ImageHelper.decode_base64_to_filename(face, 'big.png')

		if face_type in ['face', 'face_grey']:
			img = cv2.imread(image_path)
			height, width, channels = img.shape

			if height != current_app.config.get('FACE_HEIGHT') or width != current_app.config.get('FACE_WIDTH'):
				ImageHelper.minimalize_face(image_path)

		elif face_type in ['full', 'full_grey']:

			ImageHelper.minimalize(image_path_big, 250)

			big = ImageHelper.encode_base64_from_path(image_path_big)
			big = ImageHelper.decode_base64(big.decode())

			# Detection
			image_path = DetectionHelper.haar_cascade_detect(image_path)
			ImageHelper.minimalize_face(image_path)

			if image_path is None:
				return

			if Process().is_new:
				full_image_id = ImageHelper.save_image(big, 'full', g.user.id)
				ResponseParser().add_image('extraction', 'full', full_image_id)
			ImageHelper.minimalize_face(image_path)

		face = ImageHelper.encode_base64_from_path(image_path)
		face = ImageHelper.decode_base64(face.decode())

		ImageHelper.delete_image(image_path)
		ImageHelper.delete_image(image_path_big)

		return face, full_id

	@staticmethod
	def prepare_face_new(face, face_type='face'):
		# Initialize
		full_image_id = None

		# Image path for face detection
		image_path = ImageHelper.decode_base64_to_filename(face)

		# Image path for full image creation
		image_path_big = ImageHelper.decode_base64_to_filename(face, 'big.png')

		if face_type in ['full', 'full_grey']:
			# Minimize full image
			ImageHelper.minimalize(image_path_big, 250)

			# Create bytes from path
			big = ImageHelper.encode_base64_from_path(image_path_big)
			big = ImageHelper.decode_base64(big.decode())

			# Detection
			image_path = DetectionHelper.haar_cascade_detect(image_path)

			if image_path is None:
				return None, None

			ImageHelper.minimalize_face(image_path)

			if Process().is_new:
				full_image_id = ImageHelper.save_image(big, 'full', g.user.id)
				ResponseParser().add_image('extraction', 'full', full_image_id)

		if face_type in ['face', 'face_grey']:
			img = cv2.imread(image_path)
			height, width, channels = img.shape

			if height != current_app.config.get('FACE_HEIGHT') or width != current_app.config.get('FACE_WIDTH'):
				ImageHelper.minimalize_face(image_path)

		face = ImageHelper.encode_base64_from_path(image_path)
		face = ImageHelper.decode_base64(face.decode())

		ImageHelper.delete_image(image_path)
		ImageHelper.delete_image(image_path_big)

		return face, full_image_id

	@staticmethod
	def save_image(image, image_type, user_id, parent_id=None):
		image = ModelImage(
			user_id=user_id,
			image=image,
			type=image_type,
			process_id=Process().process_id,
			parent_id=parent_id
		)
		image.save()

		return image.id

	@staticmethod
	def save_numpy_image(np_image, image_type, user_id, parent_id=None):

		path = current_app.config['TEMP_PATH'] + str(time.time()) + 'tmp.png'

		cv2.imwrite(path, np_image)

		face = ImageHelper.encode_base64_from_path(path)
		face = ImageHelper.decode_base64(face.decode())

		image = ModelImage(
			user_id=user_id,
			image=face,
			type=image_type,
			process_id=Process().process_id,
			parent_id=parent_id
		)
		image.save()

		ImageHelper.delete_image(path)

		return image.id

	@staticmethod
	def save_plot_image(plt, image_type, user_id, parent_id=None):

		path = current_app.config['TEMP_PATH'] + str(time.time()) + 'tmp.png'

		plt.savefig(path)

		face = ImageHelper.encode_base64_from_path(path)
		face = ImageHelper.decode_base64(face.decode())

		image = ModelImage(
			user_id=user_id,
			image=face,
			type=image_type,
			process_id=Process().process_id,
			parent_id=parent_id
		)
		image.save()

		ImageHelper.delete_image(path)

		return image.id

	@staticmethod
	def convert_base64_to_numpy(base64face):
		base64face = str(ImageHelper.encode_base64(base64face), 'utf-8')

		decoded = base64.b64decode(base64face)

		npimg = np.fromstring(decoded, dtype=np.uint8)

		return npimg

	@staticmethod
	def convert_base64_image_to_numpy(base64face):

		decoded = base64.b64decode(base64face)

		npimg = np.fromstring(decoded, dtype=np.uint8)

		return npimg

	@staticmethod
	def adjust_gamma(image, gamma=1.0):
		# build a lookup table mapping the pixel values [0, 255] to
		# their adjusted gamma values
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
						  for i in np.arange(0, 256)]).astype("uint8")

		# apply gamma correction using the lookup table
		return cv2.LUT(image, table)
