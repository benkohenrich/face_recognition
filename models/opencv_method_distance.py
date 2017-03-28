from flask import current_app
from itsdangerous import (TimedJSONWebSignatureSerializer
						  as Serializer, BadSignature, SignatureExpired)
from passlib.apps import custom_app_context as pwd_context

from models.base import Base, db


class OpencvMethodDistance(Base):
	__tablename__ = "opencv_method_distances"

	id = db.Column(db.Integer, primary_key=True)
	code = db.Column(db.String(100))
	best = db.Column(db.INTEGER)
	worst = db.Column(db.INTEGER)