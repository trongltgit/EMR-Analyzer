import os
import logging
import zipfile
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from ydata_profiling import ProfileReport
import pandas as pd

app = Flask(__name__, static_url_path='/static')

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Biến toàn cục cho model
model = None
MODEL_PATH = "best_weights_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"

# ==========================
# TẢI MODEL NẾU CHƯA CÓ
# ==========================
def download_model():
    global model
    try:
        if not os.path.isfile(MODEL_PATH):
            logging.info("Tải model từ Google Drive...")
            r = requests.get(MODEL_URL, allow_redirects=True)
            open(MODEL_PATH, 'wb').write(r.content)
            logging.info("Model đã được tải xong!")
    except Exception as e:
        logging.error(f"Lỗi tải model: {e}")
        model = None

# ==========================
# KHỞI TẠO MODEL
# ==========================
def load_custom_model():
    global model
    if model is None:
        download_model()
        if os.path.isfile(MODEL_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                logging.info("✅ Model đã được load vào bộ nhớ!")
            except Exception as e:
                logging.error(f"Lỗi khi load model: {e}")

# ==========================
# ROUTES CHÍNH
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/emr_profile")
def emr_profile():
    return render_template("EMR_Profile.html")

@app.route("/emr_prediction")
def emr_prediction():
    return render_template("EMR_Prediction.html")

# ==========================
# XỬ LÝ FILE CSV - PROFILING
# ==========================
@app.route("/profile", methods=["POST"])
def profile():
    if 'csv_file' not in request.files:
        return "Không tìm thấy file CSV", 400

    file = request.files['csv_file']
    df = pd.read_csv(file)
    profile = ProfileReport(df, title="Hồ sơ dữ liệu EMR", explorative=True)
    output_file = os.path.join("static", "profile_report.html")
    profile.to_file(output_file=output_file)
    return render_template("EMR_Profile.html", report_path=output_file)

# ==========================
# DỰ ĐOÁN ẢNH - KERAS
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file ảnh nào được upload'}), 400

    file = request.files['image']

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Định dạng ảnh không hợp lệ!'}), 400

    try:
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        load_custom_model()

        if model is None:
            return jsonify({'error': 'Không thể tải model'}), 500

        prediction = model.predict(image_array)[0][0]
        label = 'Nodule' if prediction > 0.5 else 'Non-Nodule'

        return render_template("EMR_Prediction.html", prediction=label)

    except Exception as e:
        logging.error(f"Lỗi dự đoán: {e}")
        return jsonify({'error': 'Lỗi khi xử lý ảnh'}), 500

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Giới hạn 10MB
    app.run(debug=True)
