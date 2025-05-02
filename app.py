import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import logging
import py7zr
import glob
import gdown  # Import gdown for downloading from Google Drive

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB
CORS(app, resources={r"/*": {"origins": ["https://emr-prediction.onrender.com/", "http://localhost:3000", "http://localhost:5000"]}})

# Config model
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # Replace with your Google Drive file ID

model = None

def load_model():
    global model
    if model is None:
        try:
            # Create models directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Check if the model file exists locally
            if not os.path.exists(MODEL_PATH):
                logging.info("Model file not found locally, downloading from Google Drive...")
                try:
                    # Download the model file from Google Drive
                    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
                    logging.info("Model downloaded successfully from Google Drive.")
                except Exception as e:
                    logging.error(f"Error downloading model from Google Drive: {str(e)}")
                    raise FileNotFoundError(f"Failed to download model: {str(e)}")

            # Load the model
            logging.info("📦 Loading model into memory...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info("✅ Model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            raise

with app.app_context():
    try:
        load_model()
        logging.info("✅ Model preloaded successfully!")
    except Exception as e:
        logging.error(f"Error preloading model: {str(e)}")
        print(f"Error preloading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    logging.info("Dashboard page accessed.")
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            load_model()
            
        if 'image' not in request.files:
            logging.warning("No image file provided!")
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            logging.warning("Empty filename!")
            return jsonify({'error': 'Empty filename!'}), 400

        if not file.content_type.startswith('image/'):
            logging.warning("File is not an image!")
            return jsonify({'error': 'File must be an image (jpg, png, ...)!'}), 400

        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        logging.info(f"📊 Prediction results: {predictions}")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Error in /predict route: {str(e)}")
        print(f"Error in /predict route: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
