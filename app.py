import os
import io
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import shutil
from tensorflow.keras.models import load_model

# ========================
# CẤU HÌNH ỨNG DỤNG FLASK
# ========================
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_emr_app'

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
MODEL_NAME = 'best_weights_model.keras'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Dọn thư mục uploads nếu có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
for f in os.listdir(UPLOAD_FOLDER):
    fp = os.path.join(UPLOAD_FOLDER, f)
    try:
        if os.path.isfile(fp):
            os.remove(fp)
    except Exception as e:
        print(f"⚠️ Không thể xóa {fp}: {e}")

# ========================
# GHÉP MODEL TỰ ĐỘNG
# ========================
model_path = os.path.join(MODELS_FOLDER, MODEL_NAME)
part_files = [os.path.join(MODELS_FOLDER, f"best_weights_model.keras.{i:03d}") for i in range(1, 5)]

def merge_model_parts():
    """Ghép các phần .001–.004 thành model hoàn chỉnh"""
    if os.path.exists(model_path):
        print("✅ Model đã tồn tại, không cần ghép lại.")
        return True

    print("🧩 Đang ghép model từ các phần nhỏ...")
    try:
        with open(model_path, 'wb') as wfd:
            for part in part_files:
                if not os.path.exists(part):
                    print(f"❌ Thiếu file {part}. Dừng ghép model.")
                    return False
                with open(part, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
        print("✅ Ghép model thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

# ========================
# LOAD MODEL
# ========================
model = None
if merge_model_parts():
    try:
        model = load_model(model_path)
        print("✅ Model đã được load thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
else:
    print("⚠️ Không thể ghép model, sẽ không thể dự đoán được.")

# ========================
# CÁC HÀM TIỆN ÍCH
# ========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_emr_data(df):
    if df.empty:
        return "<h3>Báo cáo Phân tích</h3><p>Tập dữ liệu rỗng.</p>"

    if 'Diagnosis' not in df.columns:
        df['Diagnosis'] = [f'Bệnh {i % 5 + 1}' for i in range(len(df))]
    if 'Age' not in df.columns:
        df['Age'] = [25 + (i % 30) for i in range(len(df))]

    report_html = "<h3>Báo cáo Phân tích Dữ liệu EMR</h3>"
    report_html += f"<p>Tổng số hồ sơ được phân tích: <strong>{len(df)}</strong></p>"
    top_diagnoses = df['Diagnosis'].value_counts().head(5)
    report_html += "<h4>5 Chẩn đoán phổ biến nhất:</h4><table><tr><th>Chẩn đoán</th><th>Số lượng</th></tr>"
    for d, c in top_diagnoses.items():
        report_html += f"<tr><td>{d}</td><td>{c}</td></tr>"
    report_html += "</table>"
    avg_age = df['Age'].mean()
    report_html += f"<p>Tuổi trung bình của bệnh nhân: <strong>{avg_age:.1f}</strong></p>"
    return report_html

def predict_emr_image(image_path):
    if model is None:
        return "⚠️ Model chưa sẵn sàng để dự đoán."

    # Dự đoán thật (mẫu)
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image

    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        label = "Nodule (U bướu)" if preds[0][0] > 0.5 else "Non-Nodule (Không phải u bướu)"
        conf = preds[0][0] if preds[0][0] > 0.5 else 1 - preds[0][0]
        return f"Kết quả: **{label}** - Độ tin cậy: {conf * 100:.1f}%"
    except Exception as e:
        return f"❌ Lỗi khi dự đoán: {e}"

# ========================
# CÁC ROUTE
# ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    error = session.pop('error', None)
    return render_template('dashboard.html', error=error)

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    report_url, error = None, None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('emr_profile.html', error="Vui lòng chọn một file.")
        file = request.files['file']
        if not allowed_file(file.filename):
            return render_template('emr_profile.html', error="Định dạng file không hợp lệ.")

        try:
            file_content = file.read()
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'csv':
                try:
                    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(file_content.decode('latin-1')))
            else:
                df = pd.read_excel(io.BytesIO(file_content))
            report_url = analyze_emr_data(df)
        except Exception as e:
            error = f"Lỗi xử lý file: {e}"
    return render_template('emr_profile.html', report_url=report_url, error=error)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    prediction, image_path, error = None, None, None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('emr_prediction.html', error="Vui lòng chọn file hình ảnh.")
        file = request.files['file']
        if not allowed_file(file.filename):
            return render_template('emr_prediction.html', error="Định dạng file không hợp lệ.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_path = url_for('uploaded_file', filename=filename)
        prediction = predict_emr_image(filepath)
    return render_template('emr_prediction.html', prediction=prediction, image_path=image_path, error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ping_server', methods=['POST'])
def ping_server():
    return {'message': 'Server is running!'}, 200

if __name__ == '__main__':
    app.run(debug=True)
