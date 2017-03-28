import os
import base64
import PIL
import cv2
import numpy as np

from flask import current_app, g
from PIL import Image

from helpers.detectionhelper import DetectionHelper
from helpers.parsers import ResponseParser, InputParser
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

		path = current_app.config['TEMP_PATH'] + filename

		with open(path, 'wb') as f:
			f.write(imgdata)
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
	def minimalize(image_path, basewidth=None):

		if basewidth is None:
			basewidth = current_app.config.get('BASE_WIDTH')

		img = Image.open(image_path)

		width_percent = (basewidth / float(img.size[0]))

		height_size = int((float(img.size[1]) * float(width_percent)))

		# height_size = current_app.config.get('BASE_WIDTH')
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

		image_path = ImageHelper.decode_base64_to_filename(face)
		image_path_big = ImageHelper.decode_base64_to_filename(face, 'big.png')

		if face_type in ['face', 'face_grey']:
			img = cv2.imread(image_path)
			height, width, channels = img.shape

			if height != current_app.config.get('FACE_HEIGHT') or width != current_app.config.get('FACE_WIDTH'):
				ImageHelper.minimalize_face(image_path)

		elif face_type in ['full', 'full_grey']:

			ImageHelper.minimalize(image_path_big, 200)
			big = ImageHelper.encode_base64_from_path(image_path_big)
			big = ImageHelper.decode_base64(big.decode())

			if not InputParser().is_recognize:
				full_id = ImageHelper.save_image(big, 'full', g.user.id)
				ResponseParser().add_image('extraction', 'full', full_id)

			image_path = DetectionHelper.haar_cascade_detect(image_path)
			ImageHelper.minimalize_face(image_path)

		face = ImageHelper.encode_base64_from_path(image_path)
		face = ImageHelper.decode_base64(face.decode())

		ImageHelper.delete_image(image_path)
		ImageHelper.delete_image(image_path_big)

		return face

	@staticmethod
	def save_image(image, image_type, user_id):

		image = ModelImage(user_id=user_id, image=image, type=image_type, process_id=Process().process_id)
		image.save()

		return image.id

	@staticmethod
	def save_numpy_image(np_image, image_type, user_id):

		path = current_app.config['TEMP_PATH'] + 'tmp.png'

		cv2.imwrite(path, np_image)

		face = ImageHelper.encode_base64_from_path(path)
		face = ImageHelper.decode_base64(face.decode())

		image = ModelImage(user_id=user_id, image=face, type=image_type, process_id=Process().process_id)
		image.save()

		ImageHelper.delete_image(path)

		return image.id

	@staticmethod
	def save_plot_image(plt, image_type, user_id):

		path = current_app.config['TEMP_PATH'] + 'tmp.png'

		plt.savefig(path)

		face = ImageHelper.encode_base64_from_path(path)
		face = ImageHelper.decode_base64(face.decode())

		image = ModelImage(user_id=user_id, image=face, type=image_type, process_id=Process().process_id)
		image.save()

		ImageHelper.delete_image(path)

		return image.id

	@staticmethod
	def convert_base64_to_numpy(base64face):
		base64face = str(ImageHelper.encode_base64(base64face), 'utf-8')

		print(base64face)
		decoded = base64.b64decode(base64face)

		npimg = np.fromstring(decoded, dtype=np.uint8)

		return npimg

	@staticmethod
	def convert_base64_image_to_numpy(base64face):

		decoded = base64.b64decode(base64face)

		npimg = np.fromstring(decoded, dtype=np.uint8)

		return npimg
