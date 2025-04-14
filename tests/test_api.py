import unittest
import json
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_info_endpoint(self):
        response = self.app.get('/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('features', data)
        self.assertIn('classes', data)
    
    def test_predict_endpoint(self):
        # Example iris feature values (Setosa)
        test_features = [5.1, 3.5, 1.4, 0.2]
        response = self.app.post(
            '/predict',
            data=json.dumps({'features': test_features}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('class_name', data)
        self.assertIn('probabilities', data)
    
    def test_predict_with_bad_data(self):
        # Missing features
        response = self.app.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        # Wrong number of features
        response = self.app.post(
            '/predict',
            data=json.dumps({'features': [1, 2]}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
