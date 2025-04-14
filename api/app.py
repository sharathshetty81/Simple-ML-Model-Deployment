from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import sys
from werkzeug.exceptions import BadRequest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "model.joblib")
model_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "model_info.joblib")

try:
    model = joblib.load(model_path)
    model_info = joblib.load(model_info_path)
    feature_names = model_info["feature_names"]
    target_names = model_info["target_names"]
    print(f"Model loaded successfully from {model_path}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
except FileNotFoundError:
    print("Model file not found. Make sure to train the model first.")
    # For development purposes, train the model if it doesn't exist
    from model.model import train_and_save_model
    model, model_info = train_and_save_model()
    feature_names = model_info["feature_names"]
    target_names = model_info["target_names"]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Check if 'features' is in the request
        if 'features' not in data:
            return jsonify({'error': 'No features provided in the request'}), 400
        
        # Get features
        features = data['features']
        
        # Check if the number of features is correct
        if len(features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features)}',
                'expected_features': list(feature_names)
            }), 400
        
        # Convert to numpy array and reshape for a single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0].tolist()
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'class_name': target_names[prediction],
            'probabilities': {target_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}
        }
        
        return jsonify(response)
    
    except BadRequest:
        return jsonify({'error': 'Invalid JSON in request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model_type': type(model).__name__,
        'features': list(feature_names),
        'classes': list(target_names)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
