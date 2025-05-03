import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./MyDrive/efficientnet/efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
MODEL_7Z_DIR = "./models"

def extract_model_from_7z():
    if not os.path.exists(MODEL_PATH):
        logging.info("📦 Đang nối và giải nén model từ các file .7z...")
        part_files = [os.path.join(MODEL_7Z_DIR, f"best_weights_model.7z.00{i}") for i in range(1, 5)]
        full_archive = os.path.join(MODEL_7Z_DIR, "full_model.7z")

        with open(full_archive, 'wb') as f_out:
            for part in part_files:
                with open(part, 'rb') as f_in:
                    f_out.write(f_in.read())

        # Giải nén
        subprocess.run(["7z", "x", full_archive, f"-o{MODEL_DIR}"], check=True)
        logging.info("✅ Giải nén model thành công!")

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        try:
            logging.info("🌐 Đang tải model từ Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            logging.info("✅ Tải model từ Drive thành công!")
        except Exception as e:
            logging.warning(f"⚠️ Không tải được từ Drive: {e}")
            extract_model_from_7z()

model = None
def load_model():
    global model
    if model is None:
        download_model()
        logging.info("🚀 Đang load model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("✅ Model đã được load vào bộ nhớ!")

with app.app_context():
    try:
        load_model()
    except Exception as e:
        logging.error(f"Lỗi khi preload model: {e}")

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
            return jsonify({'error': 'Không có file ảnh!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)

        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Lỗi khi dự đoán: {e}")
        return jsonify({'error': f'Lỗi nội bộ: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
