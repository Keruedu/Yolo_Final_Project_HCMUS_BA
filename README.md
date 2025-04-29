# YOLO Object Detection Backend

This project is a backend application for YOLO (You Only Look Once) object detection using Flask. It allows users to upload images and receive processed images with detected objects highlighted.

## Project Structure

```
yolo-detection-backend
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── routes
│   │   ├── __init__.py
│   │   └── detection_routes.py
│   ├── models
│   │   ├── __init__.py
│   │   └── yolo_model.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── image_processing.py
│   └── static
│       └── uploads
├── tests
│   ├── __init__.py
│   ├── test_routes.py
│   └── test_model.py
├── requirements.txt
├── run.py
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd yolo-detection-backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```bash
   python run.py
   ```

2. **Upload an image:**
   Send a POST request to `/predict` with the image file.

3. **Receive the processed image:**
   The response will include the image with bounding boxes around detected objects.

## Dependencies

- Flask
- PyTorch
- OpenCV
- Other necessary libraries listed in `requirements.txt`

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.