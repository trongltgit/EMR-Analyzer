import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown
import py7zr
import logging

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Cho phép tất cả origins

# Config model
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # ID file .keras trên Google Drive
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./MyDrive/efficientnet/efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
SPLIT_FILES_DIR = "./models"  # Thư mục chứa các file nén .7z.001, .7z.002, ...

def download_model_from_drive():
    """Tải model từ Google Drive nếu chưa có."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        logging.info("🧠 Model chưa tồn tại, đang tải từ Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        logging.info("✅ Tải model từ Google Drive thành công!")

def assemble_model_from_split_files():
    """Nối các file nén .7z.001, .7z.002, ... thành file .keras."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        split_files = [os.path.join(SPLIT_FILES_DIR, f"best_weights_model.7z.{i:03d}") 
                       for i in range(1, 5)]  # Giả sử có 4 file: .001, .002, .003, .004
        if all(os.path.isfile(f) for f in split_files):
            logging.info("🧠 Tìm thấy các file nén, đang giải nén và nối...")
            archive_path = os.path.join(MODEL_DIR, "best_weights_model.7z")
            # Nối file thủ công
            with open(archive_path, 'wb') as outfile:
                for split_file in split_files:
                    with open(split_file, 'rb') as infile:
                        outfile.write(infile.read())
            # Giải nén
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                archive.extractall(path=MODEL_DIR)
            os.remove(archive_path)  # Xóa file nén tạm thời
            logging.info("✅ Đã nối và giải nén model thành công!")
        else:
            logging.warning("Không tìm thấy đủ các file nén trong thư mục models!")

def prepare_model():
    """Chuẩn bị model: ưu tiên nối file nén, nếu không thì tải từ Drive."""
    if not os.path.isfile(MODEL_PATH):
        try:
            assemble_model_from_split_files()
        except Exception as e:
            logging.error(f"Lỗi khi nối file nén: {e}")
            download_model_from_drive()

model = None
def load_model():
    global model
    if model is None:
        try:
            prepare_model()
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
        # Logic xử lý yêu cầu
        return jsonify({'classification': 'result', 'score': 0.95})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
  
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
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Lỗi trong route /predict: {e}")
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
