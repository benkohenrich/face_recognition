import os
from datetime import timedelta

root = os.path.dirname(os.path.realpath(__file__))

DEVELOPMENT_MODE = 'DEVELOPMENT'
PRODUCTION_MODE = 'PRODUCTION'
TEST_MODE = 'TEST'


class BaseConfig(object):
	DEBUG = True
	TESTING = True
	BASE_DIR = root
	SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://root@localhost/dp_1'
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	DATABASE_CONNECT_OPTIONS = {}
	THREADS_PER_PAGE = 1
	SECRET_KEY = "KeepCalmAndCarryATowel42"
	UPLOAD_FOLDER = os.path.join(root, "static/uploads")
	LOGS_FOLDER = os.path.join(root, "logs")
	SERVER_HOST = '0.0.0.0'
	TEMP_PATH = 'public/data/temp/'

class TestConfig(BaseConfig):
	DEBUG = True
	TESTING = True
	# SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://root:deprofundis12@localhost/o2_api'
	SERVER_HOST = '0.0.0.0'


class DebugConfig(BaseConfig):
	DEBUG = True
	TESTING = True
	# SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://root@localhost/o2_api'
	ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
