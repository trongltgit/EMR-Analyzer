# Import necessary libraries
import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

# Attempt to import gdown, install if not available
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'gdown'])
    import gdown

# Google Drive file ID for the TensorFlow model (replace with your actual file ID)
MODEL_FILE_ID = '1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'
MODEL_DIR = '/content/drive/MyDrive/efficientnet/efficientnet'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_weights_model.keras')

# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download the model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    drive_url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
    gdown.download(drive_url, MODEL_PATH, quiet=False)
    print("Download complete.")

# Load the TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open the image, convert to RGB and resize to 224x224
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert image to numpy array and normalize to [0,1]
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Determine class index or value
        # If model outputs a single probability (binary classification)
        if prediction.ndim == 1 or (prediction.ndim > 1 and prediction.shape[1] == 1):
            # Flatten and get first value
            pred_val = float(np.squeeze(prediction))
            class_idx = int(pred_val >= 0.5)
        else:
            # Multi-class or probability vector
            class_idx = int(np.argmax(prediction[0]))
        
        # Map class index to label
        classes = ['Non-Nodule', 'Nodule']
        result = classes[class_idx]
        
        return jsonify({'prediction': result})

    except Exception as e:
        # Return any error as JSON
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
