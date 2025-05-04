import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import py7zr  # Thư viện để giải nén .7z

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="server.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Khởi tạo Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Cho phép mọi nguồn gốc truy cập

# Đường dẫn và biến toàn cục
MODEL_DIR = "./models"
ASSEMBLED_MODEL = os.path.join(MODEL_DIR, "best_weights_model.7z")
MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.7z.{str(i).zfill(3)}")
    for i in range(1, 5)
]
model = None

# Hàm hợp nhất các tệp .7z
def assemble_model_parts():
    try:
        logging.info(f"Assembling model parts: {MODEL_PARTS}")
        with open(ASSEMBLED_MODEL, 'wb') as assembled_file:
            for part in MODEL_PARTS:
                if not os.path.exists(part):
                    logging.error(f"Missing model part: {part}")
                    raise FileNotFoundError(f"Missing part: {part}")
                with open(part, 'rb') as part_file:
                    assembled_file.write(part_file.read())
        logging.info("✅ Successfully assembled model parts into a single .7z file.")
    except Exception as e:
        logging.error(f"❌ Failed to assemble model parts: {e}")
        raise

# Hàm giải nén tệp .7z
def extract_model():
    try:
        logging.info(f"Extracting model from {ASSEMBLED_MODEL}")
        with py7zr.SevenZipFile(ASSEMBLED_MODEL, mode='r') as archive:
            archive.extractall(path=MODEL_DIR)
        logging.info("✅ Successfully extracted model .keras file.")
    except Exception as e:
        logging.error(f"❌ Failed to extract model: {e}")
        raise

# Hàm chuẩn bị mô hình
def prepare_model():
    try:
        if not os.path.exists(MODEL_PATH):
            if not os.path.exists(ASSEMBLED_MODEL):
                assemble_model_parts()
            extract_model()
    except Exception as e:
        logging.error(f"❌ Error in prepare_model: {e}")
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
    try:
        logging.info("Rendering index.html for the home route.")
        return render_template('index.html')
    except Exception as e:
        logging.error(f"❌ Error rendering home page: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

# Route dashboard
@app.route('/dashboard')
def dashboard():
    try:
        logging.info("Rendering dashboard.html for the dashboard route.")
        return render_template('dashboard.html')
    except Exception as e:
        logging.error(f"❌ Error rendering dashboard page: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

# Route kiểm tra trạng thái server
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'}), 200

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
            logging.warning("Model is not loaded in memory, attempting to load model.")
            load_model()

        # Kiểm tra file ảnh trong request
        if 'image' not in request.files:
            logging.error("No image file found in the request.")
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            logging.error("Empty filename provided in the request.")
            return jsonify({'error': 'Empty filename!'}), 400

        # Xử lý ảnh
        logging.info("Processing image for prediction...")
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()

        # Thực hiện dự đoán
        logging.info("Making prediction with the model...")
        preds = model.predict(img_array)[0][0]
        classification = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        logging.info(f"✅ Prediction successful: Class - {classification}, Score - {preds}")
        return jsonify({'classification': classification, 'score': float(preds)})

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Chạy ứng dụng
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Sử dụng cổng từ biến môi trường
    app.run(host='0.0.0.0', port=port)
