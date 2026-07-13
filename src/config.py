import os

APP_VERSION = "2.0"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER   = os.path.join(BASE_DIR, "static", "uploads")
VIDEOS_FOLDER   = os.path.join(UPLOAD_FOLDER, "videos")
MODELS_FOLDER   = os.path.join(UPLOAD_FOLDER, "models")
CAPTURES_FOLDER = os.path.join(UPLOAD_FOLDER, "captures")

DB_PATH = os.path.join(BASE_DIR, "cvvision.db")

ALLOWED_IMG   = {"png", "jpg", "jpeg", "gif", "svg", "webp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm", "m4v", "ts", "flv"}
ALLOWED_MODEL = {"pt"}

YES = "1"
NO  = "0"
