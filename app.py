import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import gdown

# Khởi tạo Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Giới hạn upload 32MB

# Đường dẫn các thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_PROFILE_REPORTS = os.path.join(BASE_DIR, 'static', 'profile_reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_FILE_ID = '1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'  # Thay bằng file ID thật nếu khác

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_PROFILE_REPORTS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Biến model toàn cục
model = None

# Tải model từ Google Drive nếu chưa có
def download_model_from_gdrive():
    try:
        print("📥 Đang tải model từ Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)
        print(f"✅ Tải model thành công: {MODEL_PATH}")
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi khi tải model từ Google Drive: {e}")
        return False

# Ghép các phần .keras.001, .keras.002,... nếu có
def merge_model_parts():
    part_files = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(MODEL_FILENAME + ".")
    ])
    if not part_files:
        print("⚠️ Không thấy các phần model.")
        return False

    try:
        with open(MODEL_PATH, 'wb') as outfile:
            for part in part_files:
                with open(os.path.join(MODELS_DIR, part), 'rb') as pf:
                    outfile.write(pf.read())
        print(f"✅ Ghép model thành công: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

# Load model
def try_load_model():
    global model

    print(f"🔍 Kiểm tra model ở: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("⚠️ File model chưa tồn tại. Thử merge các phần...")
        if not merge_model_parts():
            print("⚠️ Merge thất bại. Thử tải từ Google Drive...")
            if not download_model_from_gdrive():
                print("❌ Không thể lấy model.")
                model = None
                return

    try:
        model = load_model(MODEL_PATH)
        print("✅ Load model thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        model = None

# Load model ngay khi khởi chạy
try_load_model()

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Kiểm tra file model
@app.route("/check-model")
def check_model():
    return "✅ File tồn tại!" if os.path.exists(MODEL_PATH) else "❌ Không tìm thấy file!"

# Phân tích EMR Profile
@app.route('/emr_profile.html', methods=['GET', 'POST'])
def emr_profile():
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("emr_profile.html", error="Vui lòng chọn file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Đọc file
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                return render_template("emr_profile.html", error="File không hợp lệ (.csv, .xls, .xlsx).")

            # Giới hạn số lượng
            if df.shape[0] > 100_000 or df.shape[1] > 100:
                return render_template("emr_profile.html", error="File quá lớn (>100.000 dòng hoặc >100 cột).")

            # Tạo báo cáo
            profile = ProfileReport(df, title="EMR Report", minimal=True)
            report_path = os.path.join(STATIC_PROFILE_REPORTS, 'report.html')
            profile.to_file(report_path)
            return redirect(url_for('static', filename='profile_reports/report.html'))

        except Exception as e:
            error = f"Lỗi khi phân tích: {e}"

    return render_template("emr_profile.html", error=error)

# Dự đoán ảnh y tế
@app.route('/emr_prediction.html', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "Vui lòng chọn ảnh."
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        global model
        if model is None:
            try_load_model()
        if model is None:
            error = (
                f"Model chưa được tải hoặc không tồn tại ({MODEL_PATH}).<br>"
                "Hãy kiểm tra lại log server, đảm bảo đã upload các phần hoặc bật internet để tải model."
            )
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0) / 255.0
            pred = model.predict(img_array)
            prediction = "Nodule" if pred[0][0] > 0.5 else "Non-Nodule"
        except Exception as e:
            error = f"Lỗi khi dự đoán: {e}"

    return render_template("emr_prediction.html", prediction=prediction, error=error)

# Khởi động ứng dụng
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
