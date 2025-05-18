import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_PROFILE_REPORTS = os.path.join(BASE_DIR, 'static', 'profile_reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_PROFILE_REPORTS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

model = None

def merge_model_parts():
    """Ghép các phần .keras.001, .keras.002,... thành file .keras"""
    part_files = sorted(
        [f for f in os.listdir(MODELS_DIR) if f.startswith(MODEL_FILENAME + ".")]
    )
    if not part_files:
        print("⚠️ Không thấy các phần model.")
        return False
    print(f"🔧 Ghép model từ các phần: {part_files}")
    try:
        with open(MODEL_PATH, 'wb') as outfile:
            for part in part_files:
                with open(os.path.join(MODELS_DIR, part), 'rb') as pf:
                    while True:
                        chunk = pf.read(1024 * 1024)
                        if not chunk:
                            break
                        outfile.write(chunk)
        print(f"✅ Ghép model thành công! Đã tạo {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

def try_load_model():
    global model
    try:
        print(f"🔍 Kiểm tra model ở: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print("🔍 File model chưa tồn tại, thử merge...")
            merged = merge_model_parts()
            if not merged:
                print("⚠️ Model chưa được ghép.")
        if os.path.exists(MODEL_PATH):
            print("🔍 Đang load model...")
            model = load_model(MODEL_PATH)
            print("✅ Model đã được load.")
        else:
            print("⚠️ Không tìm thấy file model sau khi merge.")
            model = None
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        model = None

try_load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile.html', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("emr_profile.html", error="Vui lòng chọn file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                return render_template("emr_profile.html", error="File không hợp lệ (chỉ nhận .csv, .xls, .xlsx).")

            profile = ProfileReport(df, title="EMR Report", minimal=True)
            report_path = os.path.join(STATIC_PROFILE_REPORTS, 'report.html')
            profile.to_file(report_path)
            return redirect(url_for('static', filename='profile_reports/report.html'))
        except Exception as e:
            return render_template("emr_profile.html", error=f"Lỗi khi phân tích: {e}")

    return render_template("emr_profile.html")

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
            error = f"Model chưa được tải hoặc không tồn tại trên server ({MODEL_PATH}). Hãy kiểm tra lại log server."
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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
