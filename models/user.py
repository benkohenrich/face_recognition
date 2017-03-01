from flask import current_app
from itsdangerous import (TimedJSONWebSignatureSerializer
						  as Serializer, BadSignature, SignatureExpired)

from models.base import Base, db


class User(Base):
	__tablename__ = "users"

	id = db.Column(db.Integer, primary_key=True)
	email = db.Column(db.String(32), index=True)
	password = db.Column(db.String(128))
	# 	uuid = db.Column(db.String(32), nullable=False, unique=True, default=str(uuid.uuid4()))
	# 	username = db.Column(db.String(100), nullable=False, unique=True)
	# 	password = db.Column(db.String(255), nullable=False)

	def generate_auth_token(self, expiration = 600):
		s = Serializer(current_app.config['SECRET_KEY'], expires_in = expiration)
		return s.dumps({ 'id': self.id })

	@staticmethod
	def verify_auth_token(token):
		s = Serializer(current_app.config['SECRET_KEY'])
		try:
			data = s.loads(token)
		except SignatureExpired:
			return None # valid token, but expired
		except BadSignature:
			return None # invalid token
		user = User.query.get(data['id'])
		return user

#
# 	def __str__(self):
# 		return "User(uuid={})".format(self.uuid)
#
# 	def __repr__(self):
# 		return "<User>({})".format(self.uuid)
