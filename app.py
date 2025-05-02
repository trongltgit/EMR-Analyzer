import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import logging
import py7zr
import glob

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB
CORS(app, resources={r"/*": {"origins": ["https://emr-analyzer.onrender.com", "http://localhost:3000", "http://localhost:5000"]}})

# Config model
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

model = None

def load_model():
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                logging.info("Model file not found, preparing to extract...")
                # Tìm tất cả các tệp chia nhỏ
                split_files = sorted(glob.glob(os.path.join(MODEL_DIR, "best_weights_model.7z.*")))
                if not split_files:
                    logging.error("No split archive files found.")
                    raise FileNotFoundError("No split archive files found.")

                # Hợp nhất các tệp chia nhỏ thành một tệp .7z
                archive_path = os.path.join(MODEL_DIR, "best_weights_model.7z")
                with open(archive_path, 'wb') as outfile:
                    for split_file in split_files:
                        with open(split_file, 'rb') as infile:
                            outfile.write(infile.read())
                logging.info("Split archives merged successfully.")

                # Giải nén tệp .7z
                try:
                    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                        archive.extractall(path=MODEL_DIR)
                    logging.info("Model extracted successfully.")
                except Exception as e:
                    logging.error(f"Error extracting model: {str(e)}")
                    raise

                # Xóa tệp .7z sau khi giải nén (tùy chọn)
                os.remove(archive_path)

            logging.info("📦 Đang tải model vào bộ nhớ...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info("✅ Mô hình đã được load!")
        except Exception as e:
            logging.error(f"Lỗi khi tải model: {str(e)}")
            print(f"Lỗi khi tải model: {str(e)}")
            raise

with app.app_context():
    try:
        load_model()
        logging.info("✅ Mô hình đã được preload!")
    except Exception as e:
        logging.error(f"Lỗi preload model: {str(e)}")
        print(f"Lỗi preload model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    logging.info("Trang dashboard được truy cập.")
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
            logging.warning("Tên file rỗng!")
            return jsonify({'error': 'Tên file rỗng!'}), 400

        if not file.content_type.startswith('image/'):
            logging.warning("File không phải ảnh!")
            return jsonify({'error': 'File phải là ảnh (jpg, png, ...)!'}), 400

        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        logging.info(f"📊 Kết quả dự đoán: {predictions}")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Lỗi trong route /predict: {str(e)}")
        print(f"Lỗi trong route /predict: {str(e)}")
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
