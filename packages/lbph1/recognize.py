# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images("images/faces/training"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split("/")[-2])
	data.append(hist)

# Linear Support Vector Classification.
# Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
# This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.
# train a Linear SVM on the data
# C : float, optional (default=1.0)
# 		Penalty parameter C of the error term.
# loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
# 		Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
# penalty : string, ‘l1’ or ‘l2’ (default=’l2’)
# 		Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
# dual : bool, (default=True)
# 		Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
# tol : float, optional (default=1e-4)
# 		Tolerance for stopping criteria.
# multi_class: string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’) :
# 		Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.
# fit_intercept : boolean, optional (default=True)
# 		Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).
# intercept_scaling : float, optional (default=1)
# 		When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
# class_weight : {dict, ‘balanced’}, optional
# 		Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
# verbose : int, (default=0)
# 		Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.
# random_state : int seed, RandomState instance, or None (default=None)
# 		The seed of the pseudo random number generator to use when shuffling the data.
# max_iter : int, (default=1000)
# 		The maximum number of iterations to be run.
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# loop over the testing images
for imagePath in paths.list_images("images/faces/testing"):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))[0]
	# prediction = model.predict(hist)[0]

	print(prediction)
	# display the image and the prediction
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)

print ("end of calculating......")