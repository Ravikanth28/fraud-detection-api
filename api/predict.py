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

def handler(request):
    """
    Vercel serverless function handler for fraud detection.
    
    Accepts POST requests with transaction data and returns predictions.
    """
    
    # Set CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    # Handle GET request (health check)
    if request.method == 'GET':
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
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response)
        }
    
    # Handle POST request (actual predictions)
    if request.method == 'POST':
        try:
            # Parse request body
            if hasattr(request, 'body'):
                body = request.body
                if isinstance(body, bytes):
                    body = body.decode('utf-8')
                data = json.loads(body)
            else:
                data = request.json() if hasattr(request, 'json') else {}
            
            # Extract inputs
            inputs = data.get('inputs', [])
            
            if not inputs:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Missing "inputs" field in request body',
                        'expected_format': {
                            'inputs': [
                                {feat: 0.0 for feat in FEATURE_ORDER}
                            ]
                        }
                    })
                }
            
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
            response = {
                'predictions': predictions_list
            }
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(response)
            }
            
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Invalid JSON in request body'
                })
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({
                    'error': f'Prediction failed: {str(e)}'
                })
            }
    
    # Method not allowed
    return {
        'statusCode': 405,
        'headers': headers,
        'body': json.dumps({
            'error': 'Method not allowed. Use GET or POST.'
        })
    }
