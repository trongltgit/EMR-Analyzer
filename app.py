import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import Image
import gdown
import logging
from retrying import retry
import pandas as pd
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, origins=["https://emr-prediction.onrender.com"])

MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./models"  # Simpler path
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
MODEL_7Z_DIR = "./models"

model = None

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_model_gdown(url, output):
    gdown.download(url, output, quiet=False)

def assemble_model():
    global model
    try:
        small_files = ['models/best_weights_model.7z.001', 'models/best_weights_model.7z.002', 
                       'models/best_weights_model.7z.003', 'models/best_weights_model.7z.004']
        logging.info(f"Checking for .7z files: {small_files}")
        for f in small_files:
            if not os.path.exists(f):
                logging.error(f"File {f} not found")
                return
        assembled_file = 'models/best_weights_model.keras'
        
        with open(assembled_file, 'wb') as outfile:
            for small_file in small_files:
                with open(small_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        
        model = load_model(assembled_file)
        logging.info("Model loaded successfully from assembled file")
    except Exception as e:
        logging.error(f"Failed to assemble or load model: {str(e)}")
        model = None

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        try:
            logging.info("🌐 Đang tải model từ Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            download_model_gdown(url, MODEL_PATH)
            logging.info("✅ Tải model từ Drive thành công!")
        except Exception as e:
            logging.warning(f"⚠️ Không tải được từ Drive: {e}")
            assemble_model()

def load_model():
    global model
    if model is None:
        download_model()
        if model is None and os.path.isfile(MODEL_PATH):
            logging.info("🚀 Đang load model từ đường dẫn cục bộ...")
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                logging.info("✅ Model đã được load vào bộ nhớ!")
            except Exception as e:
                logging.error(f"Lỗi khi load model: {e}")

# Load model at startup
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile')
def emr_profile():
    return render_template('EMR_Profile.html')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy tệp"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên tệp không hợp lệ"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        data = pd.read_csv(filepath)
        profile = ProfileReport(data, title="Phân tích hồ sơ EMR", minimal=True)
        report_path = os.path.join(UPLOAD_FOLDER, "report_dummy.html")
        profile.to_file(report_path)

        # Sử dụng đường dẫn tương đối để frontend mở được dù ở local hay render
        return jsonify({"success": True, "report_url": request.host_url + "report"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/report", methods=["GET"])
def serve_report():
    report_path = os.path.join(UPLOAD_FOLDER, "report_dummy.html")
    if not os.path.exists(report_path):
        return "Report not found", 404
    return send_file(report_path)

@app.route("/", methods=["GET"])
def root():
    return "API hồ sơ EMR đang hoạt động."

@app.route('/emr_predictiion')
def emr_prediction():
    return render_template('EMR_Preddiction.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            load_model()
            if model is None:
                logging.error("Model failed to load")
                return jsonify({'error': 'Không thể tải model. Vui lòng thử lại sau.'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file ảnh rỗng!'}), 400

        img = Image.open(file).convert('RGB').resize((224, 224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()

        preds = model.predict(x)[0][0]
        cls = 'Nodule' if preds > 0.5 else 'Non-Nodule'
        return jsonify({'classification': cls, 'score': float(preds)})
    except Exception as e:
        logging.error(f"Lỗi khi dự đoán: {str(e)}", exc_info=True)
        return jsonify({'error': f'Lỗi xử lý ảnh hoặc dự đoán: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
