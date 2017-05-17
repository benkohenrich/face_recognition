import traceback

from flask_restful import Resource
import matplotlib.pyplot as plt
from helpers.parsers import InputParser, ErrorParser
from stats.eigenfacesstats import EigenfacesStats
from stats.fisherfacesstats import FisherfacesStats
from stats.lbpstats import LBPStats

LBP_CLASSIFICATION_ALGORITHM = {
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
	def get_stats():
		print("Check system stats")

		stats_type = InputParser().stats_type

		# Check type of the stats
		if stats_type == 'lbp':
			check_lbp_success()

			if not ErrorParser().is_empty():
				return False

			model = LBPStats(
				InputParser().__getattr__('points'),
				InputParser().__getattr__('radius'),
				InputParser().__getattr__('method'),
				InputParser().__getattr__('algorithm')
			)

			if InputParser().__getattr__('algorithm') == 'svm':
				ErrorParser().add_error('algorithm', 'SVM not implemented yet.')
				return False
			else:
				model.check_distances()

		if stats_type == 'pca':
			check_eigenfaces_success()

			if not ErrorParser().is_empty():
				return False

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

			if InputParser().__getattr__('algorithm') == 'svm':
				ErrorParser().add_error('algorithm', 'SVM not implemented yet.')
				return False
			else:
				model.check_distances()

		if stats_type == 'lda':
			check_fisherfaces_success()

			if not ErrorParser().is_empty():
				return False

			number_components = InputParser().__getattr__('number_components')

			if number_components is not None:
				number_components = int(number_components)
				if number_components == 0:
					number_components = None
				else:
					number_components = number_components

			model = FisherfacesStats(
				number_components,
				InputParser().__getattr__('tolerance'),
				InputParser().__getattr__('algorithm'),
			)

			if InputParser().__getattr__('algorithm') == 'svm':
				ErrorParser().add_error('algorithm', 'SVM not implemented yet.')
				return False
			else:
				model.check_distances()

		return True

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
	elif int(InputParser().__getattr__('number_components')) < 0:
		ErrorParser().add_error('number_components', 'extraction.number_components.must_be_positive')

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

def check_fisherfaces_success():
	if InputParser().__getattr__('number_components') is None:
		ErrorParser().add_error('number_components', 'extraction.number_components.required')
	elif int(InputParser().__getattr__('number_components')) < 0:
		ErrorParser().add_error('number_components', 'extraction.number_components.must_be_positive')

	if InputParser().__getattr__('tolerance') is None:
		ErrorParser().add_error('tolerance', 'extraction.tolerance.required')

	if InputParser().__getattr__('algorithm') is None:
		ErrorParser().add_error('algorithm', 'recognition.algorithm.required')

	if InputParser().__getattr__('algorithm') not in {
		'svm', 'euclidean', "manhattan", "chebysev", "cosine", "braycurtis"
	}:
		ErrorParser().add_error('allowed_algorithm', 'recognition.algorithm.not_allowed')

	return ErrorParser().is_empty()

