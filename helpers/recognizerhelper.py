from models.opencv_method_distance import OpencvMethodDistance
import numpy as np


class RecognizeHelper(object):
	@staticmethod
	def calculate_percentage_for_opencv_methods(method, distance, reserve=False):
		percentage = 0

		method_model = OpencvMethodDistance.query.filter(OpencvMethodDistance.code == method).first()

		if method_model is not None:
			per100 = abs(method_model.best - method_model.worst)

			percentage = (distance / per100) * 100

			print("(", distance, "/", per100, ") * 100")
			print(percentage)

			if not reserve:
				percentage = 100 - percentage

		return percentage

	@staticmethod
	def calculate_percentage_for_distance_metric_methods(user_id, recognized_distance,  distances):

		# TP = 0
		# TN = 0
		# FP = 0
		# FN = 0
		#
		only_dist = []
		for i, dist in enumerate(distances):
			only_dist.append(dist[0])

		#
		# mean = np.mean(np.asarray(only_dist))
		# print("Mean: ", mean)
		#
		# for i, dist in enumerate(distances):
		# 	if dist[0] < mean:
		# 		if int(dist[1]) == user_id:
		# 			TP += 1
		# 		else:
		# 			FP += 1
		# 	else:
		# 		if int(dist[1]) == user_id:
		# 			FN += 1
		# 		else:
		# 			TN += 1
		#
		# print("TP: ", TP)
		# # print("FN: ", FN)
		# # print("TN: ", TN)
		# print("FP: ", FP)
		#
		# TPR = TP / (TP + FP)
		#
		# print("TPR: ", TPR)
		# return TPR

		percentage = (float("{0:.5f}".format((1 - (recognized_distance / max(only_dist))) * 100)))

		# print("DISTANCE / MAX: ",percentage , "%")

		return percentage


