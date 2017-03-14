from models.base import Base, db


class AlgorithmType(Base):
	__tablename__ = "algorithm_types"

	code = db.Column(db.String(100))
	name = db.Column(db.String(255))

