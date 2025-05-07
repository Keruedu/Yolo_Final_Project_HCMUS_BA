import os
import uuid
import logging
from flask import Blueprint, request, jsonify, send_file, current_app
import cv2
from app.models.yolo_model import YOLOModel
from app.utils.image_processing import is_allowed_file, save_image, get_response_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
detection_bp = Blueprint('detection', __name__)

# Dictionary to store model instances for different model types
model_instances = {}

# Initialize default model (will be loaded when this module is imported)
default_model_type = 'yolov5'
model_instances[default_model_type] = YOLOModel(model_type=default_model_type)
# Current active model reference
model = model_instances[default_model_type]

@detection_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive an image, detect objects, and return the processed image.
    
    Returns:
        Image with detected objects or error message
    """
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        # Check if filename is empty
        if file.filename == '':
            logger.error("Empty filename submitted")
            return jsonify({'error': 'No image selected'}), 400
        
        # Get model type from request (default to yolov5 if not specified)
        model_type = request.form.get('model', default_model_type)
        logger.info(f"Using model: {model_type}")
        
        # Get or create model instance based on requested type
        global model
        if model_type not in model_instances:
            logger.info(f"Creating new model instance for type: {model_type}")
            model_instances[model_type] = YOLOModel(model_type=model_type)
        
        # Set current model to requested type
        model = model_instances[model_type]
            
        # Check if file is allowed
        if not is_allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400

        
        
        # Read the image file
        image_bytes = file.read()
        
        # Run object detection
        processed_img, results = model.detect(image_bytes)
        
        # Create a unique filename for the processed image
        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure uploads directory exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the processed image and verify it was saved successfully
        save_success = cv2.imwrite(save_path, processed_img)
        if not save_success:
            logger.error(f"Failed to save image to {save_path}")
            return jsonify({'error': 'Failed to save processed image'}), 500
            
        # Verify file exists before attempting to send it
        if not os.path.isfile(save_path):
            logger.error(f"File does not exist at {save_path}")
            return jsonify({'error': 'Failed to access processed image'}), 500
        
        # Include option to return JSON with image data
        
        if request.args.get('json_response', 'false').lower() == 'true':
            # Return image as base64 + detection results
            response_image = get_response_image(processed_img)
            response_data = {
                'image': response_image,
                'results': results
            }
            print(results)
            logger.info(f"Sending JSON response with image length: {len(response_image)} and {len(results['detections'])} detections")
            return jsonify(response_data)
        else:
            # Return the image file directly with explicit mime type and cache control
            logger.info(f"Sending file from path: {save_path}")
            return send_file(
                save_path, 
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"detected_{filename}",
                max_age=0
            )
            
    except Exception as e:
        logger.exception(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@detection_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': model.model_type}), 200