import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pickle
import cv2

# Configuration
IMG_SIZE = 224

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Feature extraction model path
SCALER_PATH = r"c:\Users\morak\CODE\ANEMIA_PROJ\scaler.pkl"
MODEL_PATH = r"c:\Users\morak\CODE\ANEMIA_PROJ\classifier_model.pkl"

def extract_features(image_array):
    """Extract features from image using histogram and edge detection."""
    try:
        # Resize to standard size
        img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Extract color histogram features
        features = []
        
        # HSV histogram (12 bins each channel = 36 features)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [12], [0, 256])
            features.extend(hist.flatten())
        
        # Color statistics
        for i in range(3):
            features.append(np.mean(hsv[:,:,i]))
            features.append(np.std(hsv[:,:,i]))
        
        # Texture features using edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.sum(edges))
        
        # Brightness and contrast
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        raise Exception(f"Feature extraction error: {str(e)}")

def load_models():
    """Load scaler and classifier if they exist."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
    except:
        pass
    
    return None, None

classifier, scaler = load_models()

def predict_image(image_path):
    """Predict anemia from image."""
    try:
        if classifier is None or scaler is None:
            return {'error': 'Model not trained yet. Please train the model first.'}
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Extract features
        features = extract_features(img_array)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = classifier.predict(features_scaled)[0]
        probability = classifier.predict_proba(features_scaled)[0]
        
        is_anemic = prediction == 1
        confidence = max(probability) * 100
        
        return {
            'prediction': 'Anemic' if is_anemic else 'Non-Anemic',
            'confidence': round(confidence, 2),
            'probability': round(max(probability), 4),
            'class_0_prob': round(probability[0], 4),  # Non-Anemic probability
            'class_1_prob': round(probability[1], 4)   # Anemic probability
        }
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

@app.route('/')
def home():
    """Render home page."""
    model_status = 'trained' if classifier is not None else 'not trained'
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp')):
        return jsonify({'error': 'Only image files allowed (JPG, PNG, GIF, BMP)'}), 400
    
    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_status = 'trained' if classifier is not None else 'not trained'
    return jsonify({
        'status': 'running',
        'model': 'Anemia Detection',
        'model_status': model_status
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 ANEMIA DETECTION - Web Application")
    print("=" * 70)
    if classifier is None:
        print("Model not trained yet!")
        print("   Please run: python train_sklearn_model.py")
    else:
        print("Model loaded successfully!")
    print()
    print(" Open your browser: http://localhost:5000")
    print(" Upload full eye or fingernail images")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    app.run(debug=False, port=5000, host='0.0.0.0')
