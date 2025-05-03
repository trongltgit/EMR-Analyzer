import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model as keras_load_model
from flask_cors import CORS
from PIL import Image
import gdown
import logging
from retrying import retry
import py7zr

# Cấu hình logging (ghi log vào file server.log)
logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)  # Cho phép tất cả origin

# Cấu hình thông số cho model
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./MyDrive/efficientnet/efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
model = None

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_model_gdown(url, output):
    gdown.download(url, output, quiet=False)

def assemble_model():
    """
    Ghép các file .7z đã chia nhỏ và giải nén để lấy file model gốc.
    """
    global model
    try:
        # Danh sách các file split trong thư mục models
        segments = [os.path.join(MODEL_DIR, f"best_weights_model.7z.00{i}") for i in range(1, 5)]
        logging.info("Kiểm tra sự tồn tại của các file split: %s", segments)
        for seg in segments:
            if not os.path.exists(seg):
                logging.error("Không tìm thấy file split: %s", seg)
                return
        # Ghép các file split thành file archive hoàn chỉnh
        assembled_archive = os.path.join(MODEL_DIR, "best_weights_model.7z")
        with open(assembled_archive, 'wb') as outfile:
            for seg in segments:
                with open(seg, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        logging.info("Ghép file .7z thành công. Bắt đầu giải nén bằng py7zr...")

        # Giải nén file archive bằng py7zr
        with py7zr.SevenZipFile(assembled_archive, mode='r') as archive:
            archive.extractall(path=MODEL_DIR)
        extracted_model = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
        model = keras_load_model(extracted_model)
        logging.info("Model được load thành công từ file giải nén!")
    except Exception as e:
        logging.error("Lỗi khi ghép hoặc load model: %s", str(e))
        model = None

def download_model():
    """
    Tải model từ Google Drive nếu chưa có. Nếu thất bại, chuyển sang ghép file .7z.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        try:
            logging.info("🌐 Đang tải model từ Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            download_model_gdown(url, MODEL_PATH)
            logging.info("✅ Tải model từ Drive thành công!")
        except Exception as e:
            logging.warning("⚠️ Không tải được model từ Drive: %s", e)
            assemble_model()

def initialize_model():
    """
    Khởi động (load) model vào bộ nhớ.
    """
    global model
    if model is None:
        download_model()
        # Nếu có file model cục bộ nhưng model chưa được load, thử load lại
        if model is None and os.path.isfile(MODEL_PATH):
            try:
                logging.info("🚀 Đang load model từ file cục bộ...")
                model = keras_load_model(MODEL_PATH)
                logging.info("✅ Model đã được load vào bộ nhớ!")
            except Exception as e:
                logging.error("Lỗi khi load model cục bộ: %s", e)

# Load model khi khởi động server
initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            initialize_model()
            if model is None:
                logging.error("Model không được load")
                return jsonify({'error': 'Không thể tải model. Vui lòng thử lại sau.'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh trong request!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file ảnh không hợp lệ!'}), 400

        # Xử lý ảnh: chuyển sang RGB và resize về 224x224
        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()
        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error("Lỗi khi thực hiện dự đoán: %s", str(e), exc_info=True)
        return jsonify({'error': f'Lỗi khi xử lý ảnh hoặc dự đoán: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
