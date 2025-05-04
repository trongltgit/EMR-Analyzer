import os
import threading
import json
import logging
import shutil
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import py7zr  # Thư viện để giải nén .7z
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import gdown

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="server.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Khởi tạo Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Cho phép mọi nguồn gốc truy cập



# File ID của tệp trên Google Drive
file_id = "your_file_id_here"  # Thay 'your_file_id_here' bằng ID của file

# Đường dẫn tải xuống file `best_weights_model.keras`
model_path = "./best_weights_model.keras"

# Tải file từ Google Drive nếu chưa tồn tại
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model file from Google Drive ID: {file_id}")
    gdown.download(url, model_path, quiet=False)
    print(f"Model file downloaded successfully to {model_path}")

# Load mô hình từ file đã tải xuống
best_model = load_model(model_path)
print("Model loaded successfully!")








# Đường dẫn và biến toàn cục
MODEL_DIR = "./models"
ASSEMBLED_MODEL = os.path.join(MODEL_DIR, "best_weights_model.7z")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.7z.{str(i).zfill(3)}")
    for i in range(1, 5)
]
model = None

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logging.info(f"📁 Created directory: {MODEL_DIR}")

# Hàm hợp nhất các tệp .7z
def assemble_model_parts():
    try:
        logging.info(f"Assembling model parts: {MODEL_PARTS}")
        with open(ASSEMBLED_MODEL, 'wb') as assembled_file:
            for part in MODEL_PARTS:
                if not os.path.exists(part):
                    logging.error(f"❌ Missing model part: {part}")
                    raise FileNotFoundError(f"Missing part: {part}")
                with open(part, 'rb') as part_file:
                    shutil.copyfileobj(part_file, assembled_file)
        logging.info("✅ Successfully assembled model parts into a single .7z file.")
    except Exception as e:
        logging.error(f"❌ Failed to assemble model parts: {e}", exc_info=True)
        raise

# Hàm giải nén tệp .7z
def extract_model():
    try:
        logging.info(f"Extracting model from {ASSEMBLED_MODEL}")
        with py7zr.SevenZipFile(ASSEMBLED_MODEL, mode='r') as archive:
            archive.extractall(path=MODEL_DIR)
        logging.info("✅ Successfully extracted model .keras file.")
    except Exception as e:
        logging.error(f"❌ Failed to extract model: {e}", exc_info=True)
        raise

# Hàm chuẩn bị mô hình
def prepare_model():
    try:
        if not os.path.exists(MODEL_PATH):  # Chỉ thực hiện nếu file .keras chưa tồn tại
            if not os.path.exists(ASSEMBLED_MODEL):
                assemble_model_parts()
            extract_model()
    except Exception as e:
        logging.error(f"❌ Error in prepare_model: {e}", exc_info=True)
        raise

# Hàm tải mô hình vào bộ nhớ
def load_model():
    global model
    try:
        prepare_model()
        logging.info("🚀 Loading model into memory...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}", exc_info=True)
        raise

# Tải mô hình khi khởi động ứng dụng
try:
    load_model()
except Exception as e:
    logging.error(f"❌ Model initialization failed: {e}")

# Route home
@app.route('/')
def home():
    return render_template('index.html')

# Route dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Route kiểm tra trạng thái server
@app.route('/ping', methods=['GET'])
def ping():
    try:
        return jsonify({'message': 'Server is running!'}), 200
    except Exception as e:
        logging.error(f"❌ Error in /ping: {e}", exc_info=True)
        return jsonify({'error': 'Server Ping Failed!'}), 500

# Route kiểm tra trạng thái mô hình
@app.route('/model-status', methods=['GET'])
def model_status():
    try:
        if model is not None:
            return jsonify({'status': 'Model is loaded successfully'}), 200
        else:
            return jsonify({'status': 'Model is not loaded'}), 503
    except Exception as e:
        logging.error(f"❌ Error checking model status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Route dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logging.warning("Model is not loaded, attempting to reload.")
            load_model()

        # Check if an image file is present in the request
        if 'image' not in request.files:
            logging.error("No image file found in the request.")
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            logging.error("Empty filename provided in the request.")
            return jsonify({'error': 'Empty filename!'}), 400

        # Process the image
        logging.info("📷 Processing image for prediction...")
        try:
            img = Image.open(file).convert('RGB').resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            img.close()
        except Exception as e:
            logging.error(f"❌ Error processing image: {e}", exc_info=True)
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        # Perform prediction
        logging.info("🤖 Making prediction...")
        try:
            preds = model.predict(img_array)[0][0]
            classification = 'Nodule' if preds > 0.5 else 'Non-Nodule'
            logging.info(f"✅ Prediction successful: Class - {classification}, Score - {preds}")
            return jsonify({'classification': classification, 'score': float(preds)})
        except Exception as e:
            logging.error(f"❌ Error during prediction: {e}", exc_info=True)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"❌ Unexpected error in /predict: {e}", exc_info=True)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Chạy ứng dụng
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Cổng mặc định là 5000 nếu biến PORT không tồn tại
    app.run(host='0.0.0.0', port=port)
