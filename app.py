import os
import secrets
import shutil
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ===============================
# 🔧 CẤU HÌNH ỨNG DỤNG
# ===============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Sinh ngẫu nhiên 1 secret key an toàn
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], 'best_weights_model.keras')

# ===============================
# 🧩 GHÉP FILE MODEL (nếu bị chia nhỏ)
# ===============================
def merge_model_parts():
    """Ghép model từ các phần .001 → .004"""
    parts = sorted([
        os.path.join(app.config['MODEL_FOLDER'], f)
        for f in os.listdir(app.config['MODEL_FOLDER'])
        if f.endswith(('.001', '.002', '.003', '.004'))
    ])
    if parts and not os.path.exists(MODEL_PATH):
        print("🔄 Đang ghép model từ các phần nhỏ...")
        with open(MODEL_PATH, 'wb') as out:
            for p in parts:
                with open(p, 'rb') as f:
                    shutil.copyfileobj(f, out)
        print("✅ Ghép model thành công:", MODEL_PATH)

# ===============================
# 🧠 TẢI MODEL DỰ ĐOÁN
# ===============================
def load_emr_model():
    merge_model_parts()
    if os.path.exists(MODEL_PATH):
        try:
            print("🧠 Đang tải model...")
            model = load_model(MODEL_PATH)
            print("✅ Model đã sẵn sàng.")
            return model
        except Exception as e:
            print("❌ Lỗi khi tải model:", e)
    else:
        print("⚠️ Model chưa tồn tại.")
    return None

model = load_emr_model()

# ===============================
# 🌐 TRANG ĐĂNG NHẬP (INDEX)
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index_page():
    """Trang đăng nhập (index.html)"""
    try:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            # 🔐 Kiểm tra đăng nhập đơn giản
            if username == 'admin' and password == '1234':
                session['logged_in'] = True
                return redirect(url_for('dashboard_page'))
            else:
                return render_template('index.html', error="Sai tài khoản hoặc mật khẩu.")

        return render_template('index.html')
    except Exception as e:
        return f"<h3>Lỗi khi tải trang đăng nhập: {e}</h3>", 500


# ===============================
# 🌟 TRANG DASHBOARD (SAU ĐĂNG NHẬP)
# ===============================
@app.route('/dashboard')
def dashboard_page():
    """Trang Dashboard chính"""
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"<h3>Lỗi khi tải dashboard.html: {e}</h3>", 500


# ===============================
# 📊 PHÂN TÍCH HỒ SƠ EMR
# ===============================
@app.route('/profile', methods=['GET', 'POST'])
def emr_profile():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    try:
        if request.method == 'POST':
            file = request.files['file']
            if not file:
                return render_template('emr_profile.html', error="Vui lòng chọn tệp.")

            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Đọc dữ liệu CSV/Excel
            if filename.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            report_html = df.describe(include='all').to_html(classes='table table-striped', border=0)
            return render_template('emr_profile.html', report_url=report_html)

        return render_template('emr_profile.html')
    except Exception as e:
        return render_template('emr_profile.html', error=f"Lỗi xử lý hồ sơ: {e}")


# ===============================
# 🤖 DỰ ĐOÁN TỪ ẢNH EMR
# ===============================
@app.route('/predict', methods=['GET', 'POST'])
def emr_prediction():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    try:
        if request.method == 'POST':
            file = request.files['file']
            if not file:
                return render_template('emr_prediction.html', error="Vui lòng chọn ảnh để phân tích.")

            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                return render_template('emr_prediction.html', error="Model chưa sẵn sàng để dự đoán.")

            # Xử lý ảnh
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Dự đoán
            preds = model.predict(img_array)
            pred_label = np.argmax(preds, axis=1)[0]

            return render_template(
                'emr_prediction.html',
                prediction=f"Kết quả dự đoán: {pred_label}",
                image_path=url_for('uploaded_file', filename=filename)
            )

        return render_template('emr_prediction.html')
    except Exception as e:
        return render_template('emr_prediction.html', error=f"Lỗi khi dự đoán: {e}")


# ===============================
# 📂 XEM FILE ĐÃ TẢI LÊN
# ===============================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===============================
# 🚪 ĐĂNG XUẤT
# ===============================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index_page'))


# ===============================
# 🚀 CHẠY ỨNG DỤNG
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
