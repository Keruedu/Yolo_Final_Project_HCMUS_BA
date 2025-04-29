# class Config:
#     DEBUG = True
#     TESTING = False
#     SECRET_KEY = 'your_secret_key_here'
#     UPLOAD_FOLDER = 'app/static/uploads'
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#     YOLO_MODEL_PATH = 'path/to/your/yolo/model'  # Update with your YOLO model path
#     YOLO_CLASSES_PATH = 'path/to/your/yolo/classes'  # Update with your YOLO classes path
#     YOLO_CONFIDENCE_THRESHOLD = 0.5
#     YOLO_NMS_THRESHOLD = 0.4

import os

class Config:
    """Configuration for the Flask application."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}