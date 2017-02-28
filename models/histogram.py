from models.base import Base, db

class Histogram(Base):
	__tablename__ = "histograms"

	image_id = db.Column(db.Integer, nullable=False)
	user_id = db.Column(db.Integer, nullable=True)
	histogram = db.Column(db.Text, nullable=False)
	gray_scale = db.Column(db.Text, nullable=True)
	number_points = db.Column(db.Integer, nullable=False, default=int(24))
	radius = db.Column(db.Integer, nullable=False, default=int(3))
	method = db.Column(db.String(64), nullable=True)

	def save(self):
		db.session.add(self)
		db.session.commit()
		db.session.flush()
