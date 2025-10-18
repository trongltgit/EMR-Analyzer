import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Cấu hình Flask
# =========================
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_emr_app'

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_weights_model.keras')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# GHÉP FILE MODEL (TỰ ĐỘNG)
# =========================
try:
    part_files = [
        os.path.join(MODEL_FOLDER, f)
        for f in sorted(os.listdir(MODEL_FOLDER))
        if f.startswith('best_weights_model.keras.') and f[-3:].isdigit()
    ]

    if part_files and not os.path.exists(MODEL_PATH):
        print("🔄 Đang ghép model từ các phần nhỏ...")
        with open(MODEL_PATH, 'wb') as outfile:
            for part in part_files:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        print("✅ Ghép model thành công:", MODEL_PATH)
    else:
        print("✅ Model đã tồn tại hoặc không tìm thấy file chia nhỏ.")
except Exception as e:
    print("⚠️ Lỗi khi ghép model:", e)

# =========================
# LOAD MODEL
# =========================
try:
    MODEL = load_model(MODEL_PATH, compile=False)
    print("✅ Model đã load thành công!")
except Exception as e:
    MODEL = None
    print(f"⚠️ Không thể load model: {e}")

# =========================
# HÀM TIỆN ÍCH
# =========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_emr_data(df):
    try:
        if df.empty:
            return "<h3>Báo cáo Phân tích</h3><p>Tập dữ liệu rỗng.</p>"

        if 'Diagnosis' not in df.columns:
            df['Diagnosis'] = [f'Bệnh {i % 5 + 1}' for i in range(len(df))]
        if 'Age' not in df.columns:
            df['Age'] = [25 + (i % 30) for i in range(len(df))]

        report_html = "<h3>Báo cáo Phân tích Dữ liệu EMR</h3>"
        report_html += f"<p>Tổng số hồ sơ: <strong>{len(df)}</strong></p>"

        top_diagnoses = df['Diagnosis'].value_counts().head(5)
        report_html += "<h4>5 Chẩn đoán phổ biến nhất:</h4><table><tr><th>Chẩn đoán</th><th>Số lượng</th></tr>"
        for d, c in top_diagnoses.items():
            report_html += f"<tr><td>{d}</td><td>{c}</td></tr>"
        report_html += "</table>"

        avg_age = df['Age'].replace([np.inf, -np.inf], np.nan).dropna().mean()
        report_html += f"<p>Tuổi trung bình: <strong>{avg_age:.1f}</strong></p>"
        return report_html
    except Exception as e:
        return f"<h3>Lỗi phân tích:</h3><p>{e}</p>"

def predict_emr_image(image_path):
    if MODEL is None:
        return "<p style='color:red;'>⚠️ Model chưa sẵn sàng để dự đoán.</p>"

    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = MODEL.predict(img_array, verbose=0)
        prob = float(pred[0][0])

        if prob >= 0.5:
            label = "Nodule (U bướu)"
            confidence = prob * 100
        else:
            label = "Non-Nodule (Không phải u bướu)"
            confidence = (1 - prob) * 100

        return f"""
        <p><strong>Kết quả:</strong> {label}</p>
        <p><strong>Độ tin cậy:</strong> {confidence:.2f}%</p>
        """
    except Exception as e:
        return f"<p>Lỗi khi dự đoán: {e}</p>"

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    error = session.pop('error', None)
    return render_template('dashboard.html', error=error)

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    report_url = None
    error = session.pop('error', None)

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = 'Vui lòng chọn file dữ liệu (.csv, .xlsx, .xls).'
            return render_template('emr_profile.html', error=error)

        if not allowed_file(file.filename):
            error = 'Định dạng file không hợp lệ.'
            return render_template('emr_profile.html', error=error)

        try:
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            data = file.read()

            if ext == 'csv':
                try:
                    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(data.decode('latin-1')))
            else:
                df = pd.read_excel(io.BytesIO(data))

            if df.empty:
                error = "File dữ liệu rỗng hoặc không đọc được."
            else:
                report_url = analyze_emr_data(df)
        except Exception as e:
            error = f"Lỗi khi xử lý file: {e}"

    return render_template('emr_profile.html', report_url=report_url, error=error)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    image_path = None
    error = session.pop('error', None)

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = 'Vui lòng chọn một file hình ảnh.'
            return render_template('emr_prediction.html', error=error)

        if not allowed_file(file.filename):
            error = 'Định dạng file không hợp lệ (chỉ jpg, jpeg, png).'
            return render_template('emr_prediction.html', error=error)

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = url_for('uploaded_file', filename=filename)
            prediction = predict_emr_image(filepath)
        except Exception as e:
            error = f"Lỗi khi xử lý hình ảnh: {e}"

    return render_template('emr_prediction.html', prediction=prediction, image_path=image_path, error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ping_server', methods=['POST'])
def ping_server():
    return {'message': 'Server is running!'}, 200

if __name__ == '__main__':
    app.run(debug=True)
