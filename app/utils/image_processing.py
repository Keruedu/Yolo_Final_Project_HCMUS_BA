# def preprocess_image(image):
#     # Function to preprocess the image for YOLO model
#     # Resize the image to the required input size
#     # Normalize pixel values
#     # Convert to the required format (e.g., tensor)
#     pass

# def postprocess_detections(detections, image_shape):
#     # Function to postprocess the detections from the YOLO model
#     # Convert bounding box coordinates to the original image scale
#     # Filter out low-confidence detections
#     # Return the processed bounding boxes and class labels
#     pass

# def save_processed_image(image, output_path):
#     # Function to save the processed image with bounding boxes
#     # Use OpenCV or PIL to save the image
#     pass

import os
import cv2
import base64

def is_allowed_file(filename, allowed_extensions):
    """
    Check if the file has an allowed extension.
    
    Args:
        filename (str): The name of the file
        allowed_extensions (set): Set of allowed file extensions
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_image(image, filepath):
    """
    Save an image to the specified filepath.
    
    Args:
        image: The image to save (numpy array)
        filepath (str): The path where to save the image
        
    Returns:
        bool: True if the image was saved successfully, False otherwise
    """
    try:
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(filepath, image)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def get_response_image(image):
    """
    Convert an image to base64 for JSON response.
    
    Args:
        image: The image to convert (numpy array)
        
    Returns:
        str: Base64 encoded image string
    """
    # Encode image as JPEG
    _, img_encoded = cv2.imencode('.jpg', image)
    # Convert to base64 string
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64