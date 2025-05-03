import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown
import logging

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

# Config model: dùng trực tiếp file .keras
MODEL_FILE_ID = "YOUR_DRIVE_FILE_ID_HERE"   # <-- thay bằng ID file .keras
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

def download_model():
    """Nếu chưa có model, tải về từ Drive."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        logging.info("🧠 Model chưa tồn tại, đang tải từ Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        # tải trực tiếp file .keras
        gdown.download(url, MODEL_PATH, quiet=False)
        logging.info("✅ Tải model thành công!")

model = None
def load_model():
    global model
    if model is None:
        try:
            download_model()
            logging.info("📦 Đang load model vào bộ nhớ...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info("✅ Mô hình đã được load!")
        except Exception as e:
            logging.error(f"Lỗi khi load model: {e}")
            raise

# Preload model khi khởi động server
with app.app_context():
    try:
        load_model()
        logging.info("✅ Mô hình đã được preload!")
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
            logging.warning("Không có file ảnh được gửi!")
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)

        preds = model.predict(x)[0][0]
        logging.info(f"📊 Kết quả dự đoán: {preds}")
        # thresholds 0.5
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Lỗi trong route /predict: {e}")
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
