import os
import logging
import numpy as np
import requests
import zipfile
import cv2
import tempfile

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image

# === Setup Flask ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Model Config ===
MODEL_DIR = "models"
MODEL_ZIP_URL = "https://raw.githubusercontent.com/trongltgit/EMR-Analyzer/main/models/model.zip"  # replace with actual URL
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, "model.zip")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")


# === Function: Download model.zip if not exists ===
def download_model_zip():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_ZIP_PATH):
        logging.info("Downloading model.zip from GitHub...")
        response = requests.get(MODEL_ZIP_URL, stream=True)
        with open(MODEL_ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Downloaded model.zip successfully.")
    else:
        logging.info("Model zip already exists.")


# === Function: Extract model zip ===
def extract_model():
    if not os.path.exists(MODEL_PATH):
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        logging.info("Extracted model successfully.")
    else:
        logging.info("Model already extracted.")


# === Load Model ===
def load_model_safe():
    try:
        download_model_zip()
        extract_model()
        model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return None


# Load model on start
best_model = load_model_safe()


# === Routes ===
@app.route('/')
def home():
    return render_template('uploader.html')


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'})


@app.route('/predict', methods=['POST'])
def predict():
    if best_model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        img = Image.open(file).convert('RGB').resize((240, 240))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = best_model.predict(img_array)
        binary_prediction = np.round(prediction).tolist()
        return jsonify({'prediction': binary_prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            image = cv2.imread(temp.name)
            image = cv2.resize(image, (240, 240))
            image = np.expand_dims(image, axis=0)

            prediction = best_model.predict(image)
            binary_prediction = np.round(prediction)

            return jsonify(binary_prediction.tolist())
    except Exception as e:
        logging.error(f"Error during OpenCV prediction: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500


# === Run App ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
