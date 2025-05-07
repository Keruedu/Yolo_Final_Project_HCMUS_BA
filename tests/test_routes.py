import unittest
import os
from app import create_app

class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.testing = True

    def test_predict_route(self):
        # Sử dụng đường dẫn tuyệt đối đến file test_image.jpg
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.isfile(test_image_path):
            self.skipTest(f"No test image found at {test_image_path}")
            
        with open(test_image_path, 'rb') as img:
            # Sửa tên tham số từ 'file' thành 'image' theo API trong detection_routes.py
            response = self.client.post('/predict', data={'image': img})
            self.assertEqual(response.status_code, 200)
            
            # Kiểm tra phản hồi dựa trên định dạng thực tế
            # Nếu là file ảnh binary
            if response.content_type == 'image/jpeg':
                self.assertTrue(len(response.data) > 0)
            # Nếu là JSON (khi sử dụng ?json_response=true)
            else:
                import json
                response_data = json.loads(response.data)
                self.assertIn('image', response_data)
                self.assertIn('results', response_data)

if __name__ == '__main__':
    unittest.main()