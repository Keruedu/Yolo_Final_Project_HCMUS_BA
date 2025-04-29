import unittest
from app.models.yolo_model import load_model, detect_objects

class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        self.model = load_model('path/to/yolo/model')

    def test_load_model(self):
        self.assertIsNotNone(self.model)

    def test_detect_objects(self):
        test_image = 'path/to/test/image.jpg'
        results = detect_objects(self.model, test_image)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # Assuming there should be at least one detection

if __name__ == '__main__':
    unittest.main()