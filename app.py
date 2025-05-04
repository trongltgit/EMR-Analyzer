import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import logging
import py7zr  # Thư viện để giải nén .7z

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Flask App Initialization
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins (adjust if needed in production)

# Constants for model management
MODEL_DIR = "./models"
ASSEMBLED_MODEL = os.path.join(MODEL_DIR, "best_weights_model.7z")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.7z.{str(i).zfill(3)}")
    for i in range(1, 5)
]

model = None  # Global model variable


# Function to assemble the split .7z files into a single file
def assemble_model_parts():
    try:
        logging.info(f"Assembling model parts: {MODEL_PARTS}")
        with open(ASSEMBLED_MODEL, 'wb') as assembled_file:
            for part in MODEL_PARTS:
                if not os.path.exists(part):
                    raise FileNotFoundError(f"Missing part: {part}")
                with open(part, 'rb') as part_file:
                    shutil.copyfileobj(part_file, assembled_file)
        logging.info("✅ Successfully assembled model parts into a single .7z file.")
    except Exception as e:
        logging.error(f"❌ Failed to assemble model parts: {e}")
        raise


# Function to extract the .7z archive to get the .keras file
def extract_model():
    try:
        logging.info(f"Extracting model from {ASSEMBLED_MODEL}")
        with py7zr.SevenZipFile(ASSEMBLED_MODEL, mode='r') as archive:
            archive.extractall(path=MODEL_DIR)
        logging.info("✅ Successfully extracted model .keras file.")
    except Exception as e:
        logging.error(f"❌ Failed to extract model: {e}")
        raise


# Function to ensure the model is ready to use
def prepare_model():
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(ASSEMBLED_MODEL):
            assemble_model_parts()
        extract_model()


# Function to load the model into memory
def load_model_into_memory():
    global model
    try:
        prepare_model()
        logging.info("🚀 Loading model into memory...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        raise


# Load model at startup
try:
    load_model_into_memory()
except Exception as e:
    logging.error(f"❌ Model initialization failed: {e}")


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
            logging.error("No image file in request.")
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            logging.error("Empty filename.")
            return jsonify({'error': 'Empty filename!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()

        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        logging.info(f"Prediction successful: {cls} (score: {preds})")
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image or prediction: {str(e)}'}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
