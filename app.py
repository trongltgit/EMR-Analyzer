from flask import Flask, render_template, request
import os
import pandas as pd
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from PIL import Image
import numpy as np
import gdown

app = Flask(__name__)

# --- Cấu hình ---
UPLOAD_FOLDER = 'uploads'
CSV_FOLDER = os.path.join(UPLOAD_FOLDER, 'csv')
IMG_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
MODEL_FOLDER = 'models'
MERGED_MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_weights_model.keras')
DRIVE_FILE_ID = '1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'  # Thay bằng ID thật nếu dùng Drive

# --- Tạo thư mục nếu chưa có ---
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# --- Hợp nhất các phần model nếu có ---
def merge_model_chunks():
    chunk_files = sorted([
        f for f in os.listdir(MODEL_FOLDER)
        if f.startswith('best_weights_model.keras') and f != 'best_weights_model.keras'
    ])
    if chunk_files:
        print("🔄 Đang hợp nhất model từ các phần:", chunk_files)
        with open(MERGED_MODEL_PATH, 'wb') as merged:
            for chunk in chunk_files:
                with open(os.path.join(MODEL_FOLDER, chunk), 'rb') as part:
                    merged.write(part.read())
        print("✅ Đã hợp nhất model")
        return True
    return False

# --- Tải model từ Google Drive nếu không có ---
def download_model_from_drive():
    if not os.path.exists(MERGED_MODEL_PATH):
        print("📥 Tải model từ Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MERGED_MODEL_PATH, quiet=False)
        print("✅ Tải model thành công")

# --- Định nghĩa lớp InputLayer tùy chỉnh để chuyển khóa cấu hình 'batch_shape'
class FixedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

# --- Monkey-patch hàm process_node để xử lý khi inbound node là chuỗi
# Chúng ta sẽ import hàm process_node từ module functional của Keras
from keras.engine import functional as keras_functional
_original_process_node = keras_functional.process_node
def patched_process_node(layer, node_data):
    # Nếu node_data là chuỗi, bỏ qua xử lý (trả về False) để tránh lỗi
    if isinstance(node_data, str):
        return False
    return _original_process_node(layer, node_data)
keras_functional.process_node = patched_process_node

# --- Load model ---
if not os.path.exists(MERGED_MODEL_PATH):
    if not merge_model_chunks():
        download_model_from_drive()

# Thiết lập custom_objects để hỗ trợ deserialization
custom_objects = {
    "Functional": tf.keras.models.Model,
    "InputLayer": FixedInputLayer,
    "DTypePolicy": Policy   # ánh xạ dtype policy
}

# Sử dụng compile=False để tránh load lại trạng thái optimizer/loss
model = tf.keras.models.load_model(MERGED_MODEL_PATH, compile=False, custom_objects=custom_objects)
print("✅ Model đã được load thành công")

# --- Trang chủ ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Dashboard ---
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# --- Phân tích hồ sơ EMR ---
@app.route('/emr-profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(CSV_FOLDER, filename)
            file.save(file_path)
            try:
                df = pd.read_csv(file_path)
                profile = ProfileReport(df, title="EMR Profile Report", explorative=True)
                report_path = os.path.join(CSV_FOLDER, 'report.html')
                profile.to_file(report_path)
                return render_template('emr_profile.html', report_url='/' + report_path)
            except Exception as e:
                return render_template('emr_profile.html', error=f"Lỗi khi tạo báo cáo: {e}")
    return render_template('emr_profile.html')

# --- Dự đoán ảnh y tế ---
@app.route('/emr-prediction', methods=['GET', 'POST'])
def emr_prediction():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(IMG_FOLDER, filename)
            file.save(file_path)
            try:
                image = Image.open(file_path).convert('RGB')
                image = image.resize((224, 224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                prediction_value = model.predict(image_array)[0][0]
                result = 'Nodule' if prediction_value >= 0.5 else 'Non-Nodule'
                return render_template('emr_prediction.html', prediction=result, image_path='/' + file_path)
            except Exception as e:
                return render_template('emr_prediction.html', error=f"Lỗi khi dự đoán: {e}")
    return render_template('emr_prediction.html')

# --- Chạy local (không dùng trên Render) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
