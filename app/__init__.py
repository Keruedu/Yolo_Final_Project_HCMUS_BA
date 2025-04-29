from flask import Flask
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Import blueprints
    from app.routes.detection_routes import detection_bp
    
    # Register blueprints
    app.register_blueprint(detection_bp)
    
    return app