import os
import threading
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import py7zr
import logging
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import urllib.request
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn đến mô hình đã được lưu trữ có độ chính xác cao nhất trên tập kiểm thử
model_path = '/content/drive/MyDrive/efficientnet/efficientnet/best_weights_model.keras'
best_model = load_model(model_path)

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


# Mo hinh moi
app = Flask(__name__)
port = "5000"

urllib.request.urlretrieve("https://raw.githubusercontent.com/trongltgit/EMR-Analyzer/refs/heads/main/templates/dashboard.html", "/content/uploader.html")

@app.route("/")
def index():
    return Path('/content/uploader.html').read_text()

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        image_path = '/content/' + file.filename
        file.save(image_path)  # Save the file to a folder named 'uploads'

        # Đọc ảnh và chuyển về kích thước mong muốn (240x240 trong trường hợp này)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (240, 240))
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch

        # Chuẩn hóa dữ liệu (nếu cần)
        # image = image / 255.0

        # Dự đoán nhãn
        prediction = best_model.predict(image)
        binary_prediction = np.round(prediction)

        return json.dumps(binary_prediction.tolist())

    return 'Error uploading file'

# end mo hinh









if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
