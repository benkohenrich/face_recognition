import os

from flask import Flask
from flask import current_app
from flask import jsonify
import base64
import PIL
from PIL import Image
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
	def encode_base64_from_path(image_path, ):
		encoded_string = ""

		with open(image_path, "rb") as image_file:
			encoded_string = base64.b64encode(image_file.read())

		return encoded_string

	@staticmethod
	def encode_base64(image_path):
		return base64.b64encode(image_path)

	@staticmethod
	def minimalize(image_path):
		print('minimalize')
		basewidth = 100
		img = Image.open(image_path)

		width_percent = (basewidth / float(img.size[0]))

		height_size = int((float(img.size[1]) * float(width_percent)))

		img = img.resize((basewidth, height_size), PIL.Image.ANTIALIAS)
		img.save(image_path)

	@staticmethod
	def delete_image(path):
		os.unlink(path)

	@staticmethod
	def prepare_face(face):
		image_path = ImageHelper.decode_base64_to_filename(face)
		ImageHelper.minimalize(image_path)
		face = ImageHelper.encode_base64_from_path(image_path)
		face = ImageHelper.decode_base64(face.decode())
		ImageHelper.delete_image(image_path)

		return face

	@staticmethod
	def save_image(image, image_type, user_id):

		image = ModelImage(user_id=user_id, image=image, type=image_type)
		image.save()

		return image.id
