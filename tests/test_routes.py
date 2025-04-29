import unittest
from app import create_app

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.testing = True

    def test_predict_route(self):
        with open('tests/test_image.jpg', 'rb') as img:
            response = self.client.post('/predict', data={'file': img})
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'processed_image', response.data)

if __name__ == '__main__':
    unittest.main()