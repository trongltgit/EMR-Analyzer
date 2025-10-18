import os
import secrets
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil

# ==============================
# CẤU HÌNH CHUNG
# ==============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg'}
ALLOWED_DATA_EXT = {'csv', 'xlsx', 'xls'}

model = None  # model toàn cục


# ==============================
# HÀM TIỆN ÍCH
# ==============================
def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def assemble_model_parts():
    """Ghép các phần .keras.001 ... .004 nếu chưa có model đầy đủ"""
    full_model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_weights_model.keras')
    if os.path.exists(full_model_path):
        print("✅ Model đã có sẵn, bỏ qua ghép.")
        return full_model_path

    parts = [
        os.path.join(app.config['MODEL_FOLDER'], f"best_weights_model.keras.{i:03d}")
        for i in range(1, 5)
    ]
    if not all(os.path.exists(p) for p in parts):
        print("⚠️ Không đủ file .001-.004 để ghép model.")
        return None

    print("🔄 Đang ghép các phần model ...")
    with open(full_model_path, 'wb') as f_out:
        for part in parts:
            with open(part, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)

    print("✅ Ghép model thành công:", full_model_path)
    return full_model_path


def load_emr_model():
    """Tải model sau khi ghép"""
    global model
    model_path = assemble_model_parts()
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("✅ Model đã được load thành công!")
        except Exception as e:
            print("❌ Lỗi khi load model:", e)
            model = None
    else:
        print("⚠️ Không thể load model vì thiếu file ghép.")


# ==============================
# ROUTE: TRANG ĐĂNG NHẬP
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'user_demo' and password == 'Test@123456':
            session['logged_in'] = True
            session['username'] = username
            print("✅ Đăng nhập thành công:", username)
            return redirect(url_for('dashboard'))
        else:
            print("❌ Sai tài khoản hoặc mật khẩu!")
            return render_template('index.html', error="Sai tên đăng nhập hoặc mật khẩu!")

    return render_template('index.html')


# ==============================
# ROUTE: DASHBOARD
# ==============================
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))
    return render_template('dashboard.html', username=session.get('username'))


# ==============================
# ROUTE: PHÂN TÍCH HỒ SƠ EMR (FILE EXCEL / CSV)
# ==============================
@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile_page():
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename, ALLOWED_DATA_EXT):
            return render_template('emr_profile.html', error="Vui lòng chọn file .csv, .xlsx hoặc .xls hợp lệ!")

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(save_path)
            else:
                df = pd.read_excel(save_path)
        except Exception as e:
            return render_template('emr_profile.html', error=f"Lỗi đọc file: {e}")

        # Thống kê đơn giản
        report_html = df.describe().to_html(classes='table table-bordered', border=0)
        return render_template('emr_profile.html', report_url=report_html)

    return render_template('emr_profile.html')


# ==============================
# ROUTE: DỰ ĐOÁN TỪ ẢNH EMR
# ==============================
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction_page():
    global model
    if not session.get('logged_in'):
        return redirect(url_for('index_page'))

    if model is None:
        load_emr_model()
        if model is None:
            return render_template('emr_prediction.html', error="⚠️ Model chưa sẵn sàng để dự đoán!")

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
            return render_template('emr_prediction.html', error="Vui lòng chọn file ảnh hợp lệ!")

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            img = image.load_img(save_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            preds = model.predict(x)

            pred_class = np.argmax(preds, axis=1)[0]
            pred_text = f"Kết quả dự đoán: Nhóm bệnh {pred_class} (Xác suất: {np.max(preds):.2%})"

            image_path = url_for('uploaded_file', filename=filename)
            return render_template('emr_prediction.html', prediction=pred_text, image_path=image_path)
        except Exception as e:
            return render_template('emr_prediction.html', error=f"Lỗi khi dự đoán: {e}")

    return render_template('emr_prediction.html')


# ==============================
# ROUTE PHỤ
# ==============================
@app.route('/uploads/<filename>')
def uploaded_file(fil_
