from flask_restful import Resource

from helpers.parsers import InputParser, ErrorParser
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

			model.check()


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

	if not ErrorParser().is_empty():
		return
