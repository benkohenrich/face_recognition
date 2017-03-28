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
	SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://heno:heno007@localhost/dp_v4'
	# SQLALCHEMY_DATABASE_URI = 'mysql+mysqldb://root@localhost/dp_v4'
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	DATABASE_CONNECT_OPTIONS = {}
	THREADS_PER_PAGE = 1
	SECRET_KEY = "KeepCalmAndCarryATowel42"
	UPLOAD_FOLDER = os.path.join(root, "static/uploads")
	LOGS_FOLDER = os.path.join(root, "logs")
	SERVER_HOST = '0.0.0.0'
	SERVER_NAME = 'http://diplserver.zbytocnosti.sk'
	TEMP_PATH = 'public/data/temp/'
	FACE_HEIGHT = 100
	FACE_WIDTH = 100
	BASE_WIDTH = 100
	IMG_RES = 100 * 100
	PREPARE_PER_USER_IMAGES = 10

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
