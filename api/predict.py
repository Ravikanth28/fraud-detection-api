from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import numpy as np

# Load model at module level (cold start optimization)
model_path = os.path.join(os.path.dirname(__file__), '..', 'fraud_model.pkl')
model = joblib.load(model_path)

# Feature order MUST match training
FEATURE_ORDER = [
    'amount', 'time_of_day', 'merchant_category',
    'distance_from_home', 'distance_from_last_transaction',
    'ratio_to_median_purchase', 'repeat_retailer',
    'used_chip', 'used_pin', 'online_order'
]

class handler(BaseHTTPRequestHandler):
    """
    Vercel serverless function handler for fraud detection.
    Uses BaseHTTPRequestHandler for compatibility with Vercel Python runtime.
    """
    
    def _set_headers(self, status_code=200):
        """Set response headers including CORS"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._set_headers(200)
        self.wfile.write(b'')
    
    def do_GET(self):
        """Handle GET request - health check"""
        try:
            # Create test prediction
            test_input = np.zeros((1, len(FEATURE_ORDER)))
            test_pred = model.predict(test_input)
            
            response = {
                'status': 'healthy',
                'message': 'Fraud Detection API is running',
                'endpoint': '/api/predict',
                'method': 'POST',
                'test_prediction': int(test_pred[0]),
                'note': 'Use POST with JSON body containing "inputs" array'
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'error': f'Health check failed: {str(e)}'
            }).encode())
    
    def do_POST(self):
        """Handle POST request - actual predictions"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            # Parse JSON
            data = json.loads(body.decode('utf-8'))
            
            # Validate input
            if 'inputs' not in data or not data['inputs']:
                self._set_headers(400)
                self.wfile.write(json.dumps({
                    'error': 'Missing "inputs" field in request body',
                    'expected_format': {
                        'inputs': [
                            {feat: 0.0 for feat in FEATURE_ORDER}
                        ]
                    }
                }).encode())
                return
            
            inputs = data['inputs']
            
            # Extract features in correct order
            features = []
            for item in inputs:
                feature_vector = [item.get(feat, 0.0) for feat in FEATURE_ORDER]
                features.append(feature_vector)
            
            # Convert to numpy array
            X = np.array(features)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Convert to list of integers
            predictions_list = [int(pred) for pred in predictions]
            
            # Return response
            self._set_headers(200)
            self.wfile.write(json.dumps({
                'predictions': predictions_list
            }).encode())
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({
                'error': 'Invalid JSON in request body'
            }).encode())
            
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'error': f'Prediction failed: {str(e)}'
            }).encode())
