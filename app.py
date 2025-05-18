import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_DIR = 'models'
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
model = None

def merge_model_parts():
    """Tự động ghép các phần .keras.001, .keras.002,... thành file .keras"""
    part_files = sorted([
        f for f in os.listdir(MODEL_DIR)
        if f.startswith(MODEL_FILENAME) and f[len(MODEL_FILENAME):].startswith(".")
    ])
    if not part_files:
        print("⚠️ Không tìm thấy các phần của model.")
        return False

    print(f"🔧 Đang ghép model từ các phần: {part_files}")
    try:
        with open(MODEL_PATH, 'wb') as outfile:
            for part in part_files:
                with open(os.path.join(MODEL_DIR, part), 'rb') as pf:
                    outfile.write(pf.read())
        print("✅ Ghép model thành công.")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

# Load model
try:
    if not os.path.exists(MODEL_PATH):
        merge_model_parts()

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model đã được load.")
    else:
        print("⚠️ Không tìm thấy file model sau khi merge.")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    model = None

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
            profile = ProfileReport(df, title="EMR Report", minimal=True)
            report_path = os.path.join('static', 'report.html')
            profile.to_file(report_path)
            return redirect(url_for('static', filename='report.html'))
        except Exception as e:
            return render_template("emr_profile.html", error=f"Lỗi khi phân tích: {e}")

    return render_template("emr_profile.html")

@app.route('/emr_prediction.html', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("emr_prediction.html", error="Vui lòng chọn ảnh.")

        if model is None:
            return render_template("emr_prediction.html", error="Model chưa được tải.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0) / 255.0
            pred = model.predict(img_array)
            prediction = "Nodule" if pred[0][0] > 0.5 else "Non-Nodule"
        except Exception as e:
            return render_template("emr_prediction.html", error=f"Lỗi khi dự đoán: {e}")

    return render_template("emr_prediction.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
