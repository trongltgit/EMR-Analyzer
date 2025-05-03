import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

# === Config ===
USE_7Z_SPLIT = True  # Nếu bạn muốn dùng file .7z.001 -> .004 thay vì Google Drive
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # Nếu dùng Drive
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

def extract_7z_parts():
    logging.info("🔧 Đang nối và giải nén các phần .7z...")
    part_files = [f"{MODEL_DIR}/best_weights_model.7z.{str(i).zfill(3)}" for i in range(1, 5)]

    for file in part_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Thiếu file: {file}")

    try:
        subprocess.run(["7z", "x", f"{part_files[0]}", f"-o{MODEL_DIR}"], check=True)
        logging.info("✅ Đã giải nén thành công .keras từ các phần .7z!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Lỗi khi giải nén: {e}")
        raise

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        if USE_7Z_SPLIT:
            logging.info("📦 Dùng file .7z chia nhỏ để lấy model...")
            extract_7z_parts()
        else:
            logging.info("📥 Đang tải model từ Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            logging.info("✅ Đã tải model thành công!")

model = None
def load_model():
    global model
    if model is None:
        try:
            download_model()
            logging.info("📦 Load model vào bộ nhớ...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info("✅ Mô hình đã được load!")
        except Exception as e:
            logging.error(f"Lỗi khi load model: {e}")
            raise

# Preload khi khởi động
with app.app_context():
    try:
        load_model()
    except Exception as e:
        logging.error(f"Lỗi preload model: {e}")

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
            load_model()

        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)

        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Lỗi trong /predict: {e}")
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
