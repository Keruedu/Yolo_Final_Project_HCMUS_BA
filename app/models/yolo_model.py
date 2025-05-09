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
        # Đường dẫn mặc định đến model đã train (có thể thay đổi trong config)
        self.trained_model_path = os.environ.get('TRAINED_MODEL_PATH', 'app/models/trained/best.pt')
        self.load_model()
        
    def load_model(self):
        """Load the YOLO model based on specified type."""
        logger.info(f"Loading {self.model_type} model...")
        try:
            if self.model_type == 'yolov5':
                # Load YOLOv5 from PyTorch Hub with the correct model name 'yolov5s'
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            elif self.model_type.startswith('yolov5') and self.model_type != 'yolov5':
                # For specific YOLOv5 variants like yolov5s, yolov5m, yolov5l, etc.
                model_size = self.model_type  # e.g., 'yolov5s', 'yolov5m'
                self.model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=True)
            elif self.model_type == 'yolov8':
                # Load YOLOv8 using ultralytics
                from ultralytics import YOLO
                self.model = YOLO('yolov8s.pt')
            elif self.model_type == 'yolov8-trained':
                # Load YOLOv8 fruits model from root directory
                from ultralytics import YOLO
                fruits_model_path = 'yolov8s_fruits.pt'
                if os.path.exists(fruits_model_path):
                    self.model = YOLO(fruits_model_path)
                    logger.info(f"Loaded fruits model from {fruits_model_path}")
                else:
                    logger.error(f"Fruits model not found at {fruits_model_path}")
                    raise FileNotFoundError(f"Fruits model file not found: {fruits_model_path}")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
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
        
        # Convert to RGB (YOLO expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        if self.model_type == 'yolov5':
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
                
        elif self.model_type in ['yolov8', 'yolov8-trained']:
            # Xử lý giống nhau cho cả yolov8 và yolov8-trained
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