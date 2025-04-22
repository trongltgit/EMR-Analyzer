import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import py7zr
import logging

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

# Config model
MODEL_DIR = "./models"
MODEL_PARTS = [
    os.path.join(MODEL_DIR, "best_weights_model.7z.001"),
    os.path.join(MODEL_DIR, "best_weights_model.7z.002"),
    os.path.join(MODEL_DIR, "best_weights_model.7z.003"),
    os.path.join(MODEL_DIR, "best_weights_model.7z.004")
]
MODEL_PATH_7Z = os.path.join(MODEL_DIR, "best_weights_model.7z")
MODEL_EXTRACTED_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

def merge_model_parts():
    if not os.path.exists(MODEL_PATH_7Z):
        logging.info("📦 Đang ghép các file .7z...")
        with open(MODEL_PATH_7Z, "wb") as output_file:
            for part in MODEL_PARTS:
                if not os.path.exists(part):
                    logging.error(f"File {part} không tồn tại!")
                    raise FileNotFoundError(f"File {part} không tồn tại!")
                with open(part, "rb") as part_file:
                    output_file.write(part_file.read())
        logging.info("✅ Ghép file .7z thành công!")

def extract_model():
    if not os.path.exists(MODEL_EXTRACTED_PATH):
        logging.info("📦 Đang giải nén model...")
        with py7zr.SevenZipFile(MODEL_PATH_7Z, mode='r') as archive:
            archive.extractall(MODEL_DIR)
        logging.info("✅ Giải nén thành công!")

model = None
def load_model():
    global model
    if model is None:
        try:
            merge_model_parts()
            extract_model()
            logging.info("📦 Đang tải model vào bộ nhớ...")
            model = tf.keras.models.load_model(MODEL_EXTRACTED_PATH)
            logging.info("✅ Mô hình đã được load!")
        except Exception as e:
            logging.error(f"Lỗi khi tải model: {str(e)}")
            raise

# Preload model khi khởi động server (nếu có thể)
with app.app_context():
    try:
        load_model()
        logging.info("✅ Mô hình đã được preload!")
    except Exception as e:
        logging.error(f"Lỗi preload model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nếu model không được preload thành công, thử tải lại
        if model is None:
            load_model()
            
        if 'image' not in request.files:
            logging.warning("Không có file ảnh được gửi!")
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        logging.info(f"📊 Kết quả dự đoán: {predictions}")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Lỗi trong route /predict: {str(e)}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
