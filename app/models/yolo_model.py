# from flask import current_app
# import torch

# class YOLOModel:
#     def __init__(self, model_path):
#         self.model = self.load_model(model_path)

#     def load_model(self, model_path):
#         # Load the YOLO model from the specified path
#         model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
#         return model

#     def detect_objects(self, image):
#         # Perform object detection on the input image
#         results = self.model(image)
#         return results.pandas().xyxy[0]  # Returns a pandas DataFrame with detections

import torch
import cv2
import numpy as np
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLO object detection model implementation."""
    
    def __init__(self, model_type='yolov5s'):
        """
        Initialize the YOLO model.
        
        Args:
            model_type (str): The type of model to load. Default is 'yolov5s'.
        """
        self.model = None
        self.model_type = model_type
        self.load_model()
        
    def load_model(self):
        """Load the YOLOv5 model from PyTorch Hub."""
        logger.info(f"Loading {self.model_type} model...")
        try:
            if 'yolov5' in self.model_type:
                # Load YOLOv5 from PyTorch Hub
                self.model = torch.hub.load('ultralytics/yolov5', self.model_type, pretrained=True)
            elif 'yolov8' in self.model_type:
                # Load YOLOv8 using ultralytics
                from ultralytics import YOLO
                self.model = YOLO(self.model_type)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def detect(self, image_bytes):
        """
        Detect objects in an image.
        
        Args:
            image_bytes (bytes): The image data as bytes.
            
        Returns:
            tuple: (processed_image, results) where processed_image is the image with bounding boxes
                  and results is a dictionary with detection details.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (YOLOv5 expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        if 'yolov5' in self.model_type:
            results = self.model(image_rgb)
            
            # Process the results
            processed_img = results.render()[0]  # Get the first image with drawn boxes
            
            # Convert back to BGR for OpenCV
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            
            # Extract detection details
            boxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes
            detection_results = {
                'num_detections': len(boxes),
                'detections': []
            }
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                detection_results['detections'].append({
                    'class': results.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
                
        elif 'yolov8' in self.model_type:
            results = self.model(image_rgb)
            
            # Process the first result (assumes batch size of 1)
            processed_img = results[0].plot()
            
            # Convert back to BGR for OpenCV
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            
            # Extract detection details
            detection_results = {
                'num_detections': len(results[0].boxes),
                'detections': []
            }
            
            for box in results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()  # get box coordinates
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                
                detection_results['detections'].append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(coord) for coord in bbox]
                })
        
        return processed_img, detection_results