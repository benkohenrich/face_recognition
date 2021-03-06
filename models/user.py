from flask import current_app, url_for
from itsdangerous import (TimedJSONWebSignatureSerializer
						  as Serializer, BadSignature, SignatureExpired)
from passlib.apps import custom_app_context as pwd_context

from models.base import Base, db


class User(Base):
	__tablename__ = "users"

	name = db.Column(db.String(100), index=True)
	password = db.Column(db.String(128))
	username = db.Column(db.String(100), nullable=False, unique=True)
	is_admin = db.Column(db.Boolean, default=False)

	def hash_password(self, password):
		self.password = pwd_context.encrypt(password)

	def verify_password(self, password):
		return pwd_context.verify(password, self.password)

	def generate_auth_token(self, expiration=60000):
		s = Serializer(current_app.config['SECRET_KEY'], expires_in=expiration)
		return s.dumps({'id': self.id})

	def summary(self):
		result = {
			'id': self.id,
			'username': self.username,
			'name': self.name,
		}

		return result

	@staticmethod
	def verify_auth_token(token):
		s = Serializer(current_app.config['SECRET_KEY'])
		try:
			data = s.loads(token)
		except SignatureExpired:
			return None  # valid token, but expired
		except BadSignature:
			return None  # invalid toke
		except:
			return None
		user = User.query.get(data['id'])
		return user
