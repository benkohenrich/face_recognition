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