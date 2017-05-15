from models.base import Base, db


# from models.image import Image


class Histogram(Base):
	__tablename__ = "histograms"

	image_id = db.Column(db.Integer, nullable=False)
	user_id = db.Column(db.Integer, nullable=True)
	histogram = db.Column(db.Text, nullable=False)
	number_points = db.Column(db.Integer, nullable=False, default=int(24))
	radius = db.Column(db.Integer, nullable=False, default=int(3))
	method = db.Column(db.String(64), nullable=True)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()

	@staticmethod
	def get_by_image_params(image_id, number_points, radius, method):
		result = Histogram.query\
			.filter(Histogram.image_id == image_id)\
			.filter(Histogram.number_points == (number_points))\
			.filter(Histogram.radius == radius)\
			.filter(Histogram.method == method).first()

		return result

	@staticmethod
	def remove_by_image(image_id):
		Histogram.query.filter(Histogram.image_id == image_id).delete()
		db.session.commit()
