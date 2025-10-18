import os
import secrets
import shutil
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# =====================================
# 🔧 CẤU HÌNH CHUNG
# =====================================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], 'best_weights_model.keras')

# =====================================
# 🔁 GHÉP MODEL TỪ NHIỀU FILE .keras.001 → .004
# =====================================
def merge_model_parts():
    parts = [f for f in sorted(os.listdir(app.config['MODEL_FOLDER'])) if f.endswith(('.001', '.002', '.003', '.004'))]
    if not parts:
        print("⚠️ Không tìm thấy file model bị chia nhỏ.")
        return

    if os.path.exists(MODEL_PATH):
        print("✅ Model đã tồn tại, bỏ qua bước ghép.")
        return

    print("🔄 Đang ghép các phần model...")
    with open(MODEL_PATH, 'wb') as outfile:
        for p in parts:
            part_path = os.path.join(app.config['MODEL_FOLDER'], p)
            with open(part_path, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    print("✅ Ghép model thành công!")

# =====================================
# ⚙️ LOAD MODEL
# =====================================
def load_emr_model():
    merge_model_parts()
    if os.path.exists(MODEL_PATH):
        try:
            print("🔍 Đang load model...")
            model = load_model(MODEL_PATH)
            print("✅ Model đã sẵn sàng.")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return None
    else:
        print("⚠️ Không tìm thấy file model.")
        return None

# Load model khi khởi động app
model = load_emr_model()

# =====================================
# 🏠 TRANG CHỦ + DASHBOARD
# =====================================
@app.route('/')
@app.route('/index.html')
def index_page():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

# =====================================
# 📄 PHÂN TÍCH HỒ SƠ EMR (CSV/XLSX)
# =====================================
@app.route('/profile', methods=['GET', 'POST'])
def emr_profile():
    error = None
    report_html = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                error = "Vui lòng chọn file để tải lên."
            else:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

                # Đọc file
                if filename.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)

                # Xử lý cơ bản: thống kê mô tả
                desc = df.describe(include='all').fillna('')
                report_html = "<h3>Báo cáo thống kê mô tả</h3>" + desc.to_html(classes='table table-bordered')

        except Exception as e:
            error = f"Lỗi xử lý file: {e}"

    return render_template('emr_profile.html', error=error, report_url=report_html)

# =====================================
# 🧠 DỰ ĐOÁN HÌNH ẢNH EMR
# =====================================
@app.route('/predict', methods=['GET', 'POST'])
def emr_prediction():
    error = None
    prediction = None
    image_path = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                error = "Vui lòng tải lên hình ảnh hợp lệ."
            else:
                filename = file.filename
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                image_path = url_for('uploaded_file', filename=filename)

                if model is None:
                    error = "Model chưa sẵn sàng. Vui lòng kiểm tra lại."
                else:
                    # Giả sử model nhận input kích thước 224x224
                    img = image.load_img(path, target_size=(224, 224))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    preds = model.predict(img_array)

                    pred_label = np.argmax(preds, axis=1)[0]
                    prediction = f"Kết quả dự đoán: <strong>{pred_label}</strong>"

        except Exception as e:
            error = f"Lỗi xử lý ảnh: {e}"

    return render_template('emr_prediction.html', error=error, prediction=prediction, image_path=image_path)

# =====================================
# 🖼️ PHỤ TRỢ HIỂN THỊ FILE ẢNH UPLOAD
# =====================================
from flask import url_for
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# =====================================
# 🚀 CHẠY APP
# =====================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
