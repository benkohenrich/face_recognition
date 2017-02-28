# import uuid as uuid
# from models.base import Base, db
#
#
# class User(Base):
# 	__tablename__ = "users"
#
#
# 	uuid = db.Column(db.String(32), nullable=False, unique=True, default=str(uuid.uuid4()))
# 	username = db.Column(db.String(100), nullable=False, unique=True)
# 	password = db.Column(db.String(255), nullable=False)
#
# 	def __str__(self):
# 		return "User(uuid={})".format(self.uuid)
#
# 	def __repr__(self):
# 		return "<User>({})".format(self.uuid)
