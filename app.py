import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import gdown
import logging
from retrying import retry

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App Initialization
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins (adjust if needed in production)

# Constants for model management
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./content/drive/MyDrive/efficientnet/efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

model = None  # Global model variable


# Function to retry downloading the model
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_model_gdown(url, output):
    gdown.download(url, output, quiet=False)


# Function to ensure the model is downloaded and loaded
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        try:
            logging.info("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            download_model_gdown(url, MODEL_PATH)
            logging.info("Model downloaded successfully!")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise


# Function to load the model into memory
def load_model_into_memory():
    global model
    if model is None:
        try:
            download_model()
            logging.info("Loading model from local path...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise


# Load model at startup
try:
    load_model_into_memory()
except Exception as e:
    logging.error(f"Model initialization failed: {e}")


# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            load_model_into_memory()

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()

        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image or prediction: {str(e)}'}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
