from flask import Flask, render_template, request
import os
import pandas as pd
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown

app = Flask(__name__)

# --- Cấu hình ---
UPLOAD_FOLDER = 'uploads'
CSV_FOLDER = os.path.join(UPLOAD_FOLDER, 'csv')
IMG_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
MERGED_MODEL_PATH = 'models/best_weights_model.keras'
DRIVE_FILE_ID = '1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'  # Thay bằng real ID Google Drive

# Tạo thư mục nếu chưa có
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Hợp nhất model nếu bị chia nhỏ ---
def merge_model_chunks():
    chunk_files = sorted([
        f for f in os.listdir('models')
        if f.startswith('best_weights_model.keras') and not f.endswith('.keras')
    ])
    if chunk_files:
        print("🔄 Hợp nhất model từ các phần:", chunk_files)
        with open(MERGED_MODEL_PATH, 'wb') as merged:
            for chunk in chunk_files:
                with open(os.path.join('models', chunk), 'rb') as part:
                    merged.write(part.read())
        print("✅ Hợp nhất model thành công")
        return True
    return False

# --- Tải model từ Drive nếu cần ---
def download_model_from_drive():
    if not os.path.exists(MERGED_MODEL_PATH):
        print("📥 Tải model từ Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MERGED_MODEL_PATH, quiet=False)
        print("✅ Tải model thành công")

# --- Load model ---
if not os.path.exists(MERGED_MODEL_PATH):
    if not merge_model_chunks():
        download_model_from_drive()

model = tf.keras.models.load_model(MERGED_MODEL_PATH)
print("✅ Model đã load thành công")



# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# --- Phân tích hồ sơ EMR (CSV) ---
@app.route('/emr-profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(CSV_FOLDER, filename)
            file.save(file_path)

            df = pd.read_csv(file_path)
            profile = ProfileReport(df, title="EMR Profile Report", explorative=True)
            report_path = os.path.join(CSV_FOLDER, 'report.html')
            profile.to_file(report_path)
            return render_template('emr_profile.html', report_url='/' + report_path)
    return render_template('emr_profile.html')

# --- Dự đoán ảnh y tế ---
@app.route('/emr-prediction', methods=['GET', 'POST'])
def emr_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(IMG_FOLDER, filename)
            file.save(file_path)

            # Xử lý ảnh
            image = Image.open(file_path).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Dự đoán
            prediction = model.predict(image_array)[0][0]
            result = 'Nodule' if prediction >= 0.5 else 'Non-Nodule'

            return render_template('emr_prediction.html', result=result, image_path='/' + file_path)
    return render_template('emr_prediction.html')

# --- Chạy app ---
if __name__ == '__main__':
    app.run(debug=True)
