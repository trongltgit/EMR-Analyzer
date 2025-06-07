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
import zipfile

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

# --- Kiểm tra file model có hợp lệ hay không (dạng zip) ---
def is_valid_keras_file(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r'):
            return True
    except zipfile.BadZipFile:
        print(f"❌ File {file_path} không phải là file .keras hợp lệ.")
        return False

# --- Tải model từ Google Drive nếu không có hoặc không hợp lệ ---
def download_model_from_drive():
    if not os.path.exists(MERGED_MODEL_PATH) or not is_valid_keras_file(MERGED_MODEL_PATH):
        print("📥 Tải model từ Google Drive...")
        try:
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MERGED_MODEL_PATH, quiet=False)
            print("✅ Tải model thành công")
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            raise

# --- Định nghĩa lớp InputLayer tùy chỉnh để chuyển khóa cấu hình 'batch_shape' ---
class FixedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

# --- Load model ---
download_model_from_drive()

# In gỡ lỗi để kiểm tra đường dẫn
print("Thư mục làm việc hiện tại:", os.getcwd())
print("Đường dẫn tuyệt đối của file model:", os.path.abspath(MERGED_MODEL_PATH))

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
