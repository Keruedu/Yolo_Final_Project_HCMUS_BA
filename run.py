from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.routes.detection_routes import detection_bp

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    
    app.register_blueprint(detection_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)