from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
import os
import io
import secrets
import shutil
import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FOLDER = 'models'
MERGED_MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_weights_model_merged.keras')

# =============================
# Ghép model .keras.001–.004
# =============================
def merge_model_files():
    parts = [os.path.join(MODEL_FOLDER, f"best_weights_model.keras.{i:03d}") for i in range(1, 5)]
    if not all(os.path.exists(p) for p in parts):
        print("⚠️ Thiếu file model (.001–.004)")
        return None
    if os.path.exists(MERGED_MODEL_PATH):
        return MERGED_MODEL_PATH

    print("🔧 Đang ghép các phần model...")
    with open(MERGED_MODEL_PATH, "wb") as merged:
        for part in parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, merged)
    print("✅ Đã ghép xong model.")
    return MERGED_MODEL_PATH


# =============================
# Load model
# =============================
MODEL_PATH = merge_model_files()
model = None
if MODEL_PATH:
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model y tế đã load thành công.")
    except Exception as e:
        print("❌ Lỗi khi load model:", e)


# =============================
# Trang chủ (index)
# =============================
@app.route('/')
def index():
    return render_template('index.html')


# =============================
# Dashboard
# =============================
@app.route('/dashboard')
def dashboard():
    # Nếu chưa đăng nhập, nhưng đến từ JS của index.html => cho đăng nhập tự động
    if not session.get('logged_in'):
        session['logged_in'] = True
    return render_template('dashboard.html')


# =============================
# Trang phân tích hồ sơ EMR
# =============================
@app.route('/emr_profile')
def emr_profile():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template('emr_profile.html')


# =============================
# Xử lý upload hồ sơ EMR
# =============================
@app.route('/upload_emr', methods=['POST'])
def upload_emr():
    if not session.get('logged_in'):
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('Không có file nào được tải lên.')
        return redirect(url_for('emr_profile'))

    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn file hợp lệ.')
        return redirect(url_for('emr_profile'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
        summary = df.describe(include='all').to_html(classes='table table-bordered table-sm')
        flash('✅ Phân tích hồ sơ EMR thành công!')
    except Exception as e:
        summary = f"Lỗi khi đọc file: {e}"
        flash('❌ Lỗi khi phân tích hồ sơ.')

    return render_template('emr_profile.html', summary=summary, filename=file.filename)


# =============================
# Trang phân tích ảnh y tế
# =============================
@app.route('/emr_prediction')
def emr_prediction():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template('emr_prediction.html')


# =============================
# Upload ảnh & dự đoán
# =============================
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if not session.get('logged_in'):
        return redirect(url_for('index'))

    if 'image' not in request.files:
        flash('Không có ảnh nào được tải lên.')
        return redirect(url_for('emr_prediction'))

    file = request.files['image']
    if file.filename == '':
        flash('Chưa chọn ảnh hợp lệ.')
        return redirect(url_for('emr_prediction'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert('RGB').resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        result = float(model.predict(arr)[0][0]) if model else None
    except Exception as e:
        print("❌ Lỗi khi dự đoán:", e)
        result = None

    return render_template('emr_prediction.html', image_name=file.filename, result=result)


# =============================
# Đăng xuất
# =============================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# =============================
# Chạy app
# =============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
