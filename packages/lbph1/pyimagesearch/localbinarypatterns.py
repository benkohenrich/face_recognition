# import the necessary packages
from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints = 24, radius = 8):

		if numPoints == None or numPoints == "":
			numPoints = 24

		if radius == None or radius == "":
			radius = 8

		self.numPoints = int(numPoints)
		self.radius = int(radius)

		print(self.radius)
		print(self.numPoints)

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns

		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns

		# print (hist);
		return hist