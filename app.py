import os
import secrets
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model

# ==============================
# CẤU HÌNH ỨNG DỤNG
# ==============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# ==============================
# GHÉP MODEL .KERAS NẾU CẦN
# ==============================
model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_weights_model.keras')

if not os.path.exists(model_path):
    print("🔄 Đang ghép các phần của model...")
    parts = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.startswith('best_weights_model.keras.')]
    parts.sort()

    if parts:
        with open(model_path, 'wb') as output_file:
            for part in parts:
                with open(os.path.join(app.config['MODEL_FOLDER'], part), 'rb') as pf:
                    output_file.write(pf.read())
        print("✅ Ghép model thành công!")
    else:
        print("⚠️ Không tìm thấy các phần của model, vui lòng kiểm tra thư mục models/.")

# ==============================
# TẢI MODEL (NẾU CÓ)
# ==============================
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print("✅ Model đã được tải thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
else:
    print("⚠️ Chưa có model đầy đủ để tải.")

# ==============================
# ROUTES
# ==============================

# Trang Đăng nhập
@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Tài khoản hợp lệ
        if username == 'user_demo' and password == 'Test@123456':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('index.html', error="Sai tên đăng nhập hoặc mật khẩu!")

    return render_template('index.html')


# Trang Dashboard sau khi đăng nhập
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    return render_template('dashboard.html', username=session.get('username'))


# Trang Phân tích hồ sơ bệnh nhân
@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("Vui lòng chọn file trước khi tải lên!", "error")
            return redirect(url_for('profile_page'))
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        flash("Tải file thành công!", "success")
    return render_template('emr_profile.html')


# Trang Dự đoán bệnh từ dữ liệu / hình ảnh
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    prediction_result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if model:
                # Mô phỏng dự đoán (vì chưa biết input thực tế)
                prediction_result = "Kết quả dự đoán: Bệnh nhân có nguy cơ thấp."
            else:
                prediction_result = "⚠️ Model chưa sẵn sàng!"
    return render_template('emr_prediction.html', result=prediction_result)


# Đăng xuất
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index_page'))


# ==============================
# KHỞI CHẠY ỨNG DỤNG
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
