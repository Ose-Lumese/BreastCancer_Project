from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler from the /model/ folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✓ Model and Scaler loaded successfully")
except Exception as e:
    print(f"✗ Error loading model files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 5 features as selected in Part A
        features = np.array([[
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean'])
        ]])
        
        # 1. Scale the input using the saved scaler
        scaled_features = scaler.transform(features)
        
        # 2. Predict
        prediction = model.predict(scaled_features)[0]
        
        # 3. Map result (If Malignant=1, Benign=0 in your training)
        result = "Malignant" if prediction == 1 else "Benign"
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)