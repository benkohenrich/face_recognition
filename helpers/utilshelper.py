import numpy as np

from helpers.parsers import ErrorParser, ResponseParser, InputParser


class Utils(object):
	@staticmethod
	def remove_duplicates(values):
		output = []
		seen = set()
		for value in values:
			# If value has not been encountered yet,
			# ... add it to both list and set.
			if value not in seen:
				output.append(value)
				seen.add(value)
		return output

	@staticmethod
	def calculate_percentage_from_distances(distances, distance, reserve=False):
		print(distances)
		print(distance)

		only_distances = []
		# print(np.sum(distances))
		for j, dist in enumerate(distances):
			only_distances.append(dist[0])

		print("SUM:", np.sum(only_distances))

		for j, dist in enumerate(distances):
			print(float("{0:.20f}".format(dist[0])))
			print("User no.", dist[1], " : ", (float("{0:.5f}".format(dist[0])) / np.sum(only_distances)) * 100, " %")

	@staticmethod
	def reset_singletons():
		ErrorParser().reset()
		InputParser().reset()
		ResponseParser().reset()
