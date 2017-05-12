from flask_restful import Resource
import matplotlib.pyplot as plt
from helpers.parsers import InputParser, ErrorParser
from stats.eigenfacesstats import EigenfacesStats
from stats.lbpstats import LBPStats

LBP_CLASSIFICATION_ALGORITHM = {
	"svm",
	"correlation",
	"chi-squared",
	"intersection",
	"bhattacharyya",
	"euclidean",
	"manhattan",
	"chebysev",
	"cosine",
	"braycurtis",
}


class Stats(Resource):
	@staticmethod
	def statistics():
		# X = [ 0, 0.5 , 0.9 , 1 , 1 ,1 ,1]
		# Y = [ 0, 0.5 , 0.9 , 1 , 1 ,1 ,1]

		# plt.figure()
		# label = "intersection \ncorrelation\nchi-squared\nbhattacharyya "
		# print("plot save")
		# plt.plot(X, Y, label=label)
		# plt.legend(loc='lower right')
		# plt.plot([0, 1], [0, 1], 'k--')
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.savefig("text.jpg")
		# check_lbp_success()
		# r = [ 8, 10, 12 ]
		# p = [ 24, 30, 36 ]
		for algorithm in ["manhattan", "chebysev", "cosine"]:
			# print(algorithm)
			# for i ,radius in enumerate(r):
		# 		points = p[i]
		# 	for method in ['randomized']:
			for method in ['auto', 'full', 'randomized']:
				print(algorithm, '-', method)
				number_components = InputParser().__getattr__('number_components')

				if number_components is not None:
					number_components = int(number_components)
					if number_components == 0:
						number_components = None
					else:
						number_components = number_components

				model = EigenfacesStats(
					50,
					method,
					algorithm,
					InputParser().__getattr__('whiten'),
				)

				try:
					model.check_distances()
				except:
					print("Fault")
		#
		# 			generate = False
		#
		# 			# if algorithm == 'chi-squared':
		# 			# 	generate = True
		# 			# if algorithm == 'chebysev' and radius == 8 and points == 24 and method in [ 'nri_uniform', 'var' ]:
		# 			# 	generate = True
		# 			# if algorithm == 'chebysev' and radius == 10 and points == 30 and method in ['default', 'ror', 'uniform']:
		# 			# 	generate = True
		# 			# if algorithm == 'euclidean' and radius == 12 and points == 36 and method in [ 'var' ]:
		# 			# 	generate = True
		#
		# 			if algorithm == 'cosine' and radius == 8 and points == 24 and method in ['uniform']:
		# 				generate = True
		# 			if algorithm == 'cosine' and radius == 10 and points == 30 and method in ['ror',
		# 																						'uniform']:
		# 				generate = True
		# 			if algorithm == 'cosine' and radius == 12 and points == 36 and method in [ 'ror', 'uniform','var' ]:
		# 				generate = True
		#
		# 			if algorithm == 'braycurtis' and radius == 8 and points == 24 and method in ['nri_uniform',
		# 																					 'var']:
		# 				generate = True
		# 			# if algorithm == 'braycurtis' and radius == 10 and points == 30 and method in ['default',
		# 			# 																		  'uniform', 'var']:
		# 			# 	generate = True
		# 			# if algorithm == 'braycurtis' and radius == 12 and points == 36 and method in [ 'var']:
		# 			# 	generate = True
		# 			if generate:
		# 				print("Calculate: algorithm=", algorithm, " radius=", str(radius), " points=", str(points), " method=", method)
		# 				InputParser().extraction_settings = {
		# 					'method': method,
		# 					'points': points,
		# 					'radius':radius
		# 				}
		# 				InputParser().recognition_settings = {
		# 					'algorithm': algorithm
		# 				}
		# 				model = LBPStats(
		# 					points,
		# 					radius,
		# 					method,
		# 					algorithm
		# 				)

						# model.check2()

	@staticmethod
	def get_stats():
		print("Check system stats")

		stats_type = InputParser().stats_type

		# Check type of the stats
		if stats_type == 'lbp':
			check_lbp_success()
			model = LBPStats(
				InputParser().__getattr__('points'),
				InputParser().__getattr__('radius'),
				InputParser().__getattr__('method'),
				InputParser().__getattr__('algorithm')
			)

			if InputParser().__getattr__('algorithm') == 'svm':
				model.check_svm()
			else:
				model.check_distances()

		if stats_type == 'pca':
			check_eigenfaces_success()

			number_components = InputParser().__getattr__('number_components')

			if number_components is not None:
				number_components = int(number_components)
				if number_components == 0:
					number_components = None
				else:
					number_components = number_components

			model = EigenfacesStats(
				number_components,
				InputParser().__getattr__('method'),
				InputParser().__getattr__('algorithm'),
				InputParser().__getattr__('whiten'),
			)

			model.check_distances()


def check_lbp_success():
	if InputParser().__getattr__('points') is None:
		ErrorParser().add_error('points', 'extraction.points.required')

	if InputParser().__getattr__('radius') is None:
		ErrorParser().add_error('radius', 'extraction.radius.required')

	if InputParser().__getattr__('method') is None:
		ErrorParser().add_error('method', 'extraction.method.required')

	if InputParser().__getattr__('method') not in {'default', 'ror', 'uniform', 'nri_uniform', 'var'}:
		ErrorParser().add_error('method_allowed', 'extraction.method.not_allowed')

	if InputParser().__getattr__('algorithm') is None:
		ErrorParser().add_error('algorithm', 'recognition.algorithm.required')

	if InputParser().__getattr__('algorithm') not in LBP_CLASSIFICATION_ALGORITHM:
		ErrorParser().add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

	return ErrorParser().is_empty()


def check_eigenfaces_success():
	if InputParser().__getattr__('number_components') is None:
		ErrorParser().add_error('number_components', 'extraction.number_components.required')

	if InputParser().__getattr__('method') is None:
		ErrorParser().add_error('method', 'extraction.method.required')

	if InputParser().__getattr__('method') not in {'auto', 'full', 'randomized'}:
		ErrorParser().add_error('method_allowed', 'extraction.method.not_allowed')

	if InputParser().__getattr__('algorithm') is None:
		ErrorParser().add_error('algorithm', 'recognition.algorithm.required')

	if InputParser().__getattr__('algorithm') not in {
		'svm', 'euclidean', "manhattan", "chebysev", "cosine", "braycurtis"
	}:
		ErrorParser().add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

	return ErrorParser().is_empty()
