import unittest
import os
from app.models.yolo_model import YOLOModel

class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        self.model = YOLOModel()

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_detect_on_sample_image(self):
        # Đảm bảo có một file test_image.jpg trong thư mục tests
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if not os.path.isfile(test_image_path):
            self.skipTest("No test_image.jpg found in tests/ folder")
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        processed_img, results = self.model.detect(image_bytes)
        self.assertIsNotNone(processed_img)
        self.assertIn("detections", results)

if __name__ == '__main__':
    unittest.main()