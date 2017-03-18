from flask import current_app
from itsdangerous import (TimedJSONWebSignatureSerializer
						  as Serializer, BadSignature, SignatureExpired)
from passlib.apps import custom_app_context as pwd_context

from models.base import Base, db


class User(Base):
	__tablename__ = "users"

	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(100), index=True)
	password = db.Column(db.String(128))
	username = db.Column(db.String(100), nullable=False, unique=True)
	original_image_id = db.Column(db.INTEGER, index=True, nullable=True)

	def hash_password(self, password):
		self.password = pwd_context.encrypt(password)

	def verify_password(self, password):
		return pwd_context.verify(password, self.password)

	def generate_auth_token(self, expiration = 60000):
		s = Serializer(current_app.config['SECRET_KEY'], expires_in = expiration)
		return s.dumps({'id': self.id})

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
