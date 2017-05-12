import cv2

from helpers.parsers import ErrorParser


class DetectionHelper(object):
	@staticmethod
	def haar_cascade_detect(filepath):

		found_face = []

		face_cascade = cv2.CascadeClassifier(
			'public/haarcascade_frontalface_default.xml')

		img = cv2.imread(filepath)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		for scale in [ float(i) / 10 for i in range(11, 15) ]:
			for neighbors in range(2, 5):
				rects = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
				if len(rects) == 1:
					found_face = rects

		faces = found_face

		if len(faces) == 0:
			ErrorParser().add_error('detection', "extraction.detection.no_face")
			return None
		elif len(faces) > 1:
			ErrorParser().add_error('detection', "extraction.detection.more_faces")
			return None

		for (x, y, w, h) in faces:

			distance_x = 10
			distance_y = 20

			if x < distance_x:
				distance_x = x
			if y < distance_y:
				distance_y = y

			x -= distance_x
			y -= distance_y

			h += distance_y
			w += distance_x
			crop_img = img[y:(y + h), x:(x + w)]
			cv2.imwrite(filepath, crop_img)

		return filepath
