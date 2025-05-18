import os
from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import uuid
from ydata_profiling import ProfileReport
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# === Cấu hình ===
MODEL_DIR = 'models'
MERGED_MODEL_PATH = os.path.join(MODEL_DIR, 'best_weights_model.keras')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ghép model nếu cần
if not os.path.exists(MERGED_MODEL_PATH):
    try:
        with open(MERGED_MODEL_PATH, 'wb') as output_file:
            for i in range(1, 5):
                part_filename = os.path.join(MODEL_DIR, f'best_weights_model.keras.00{i}')
                with open(part_filename, 'rb') as part:
                    output_file.write(part.read())
        print("✅ Model merged successfully.")
    except Exception as e:
        print(f"❌ Model merge failed: {e}")

# Load model
model = None
try:
    if os.path.exists(MERGED_MODEL_PATH):
        model = load_model(MERGED_MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print("⚠️ Model file not found.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# === ROUTES ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# === Trang 1: Phân tích hồ sơ EMR (CSV) ===
@app.route("/emr_profile.html", methods=["GET", "POST"])
def emr_profile():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("emr_profile.html", error="Vui lòng chọn file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
            report_path = os.path.join("static", "report.html")
            profile = ProfileReport(df, title="EMR Report", minimal=True)
            profile.to_file(report_path)
            return redirect(url_for('static', filename='report.html'))
        except Exception as e:
            return render_template("emr_profile.html", error=f"Lỗi khi phân tích file: {e}")

    return render_template("emr_profile.html")

# === Trang 2: Dự đoán ảnh EMR ===
@app.route("/emr_prediction.html", methods=["GET", "POST"])
def emr_prediction():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("emr_prediction.html", error="Vui lòng chọn ảnh.")

        if model is None:
            return render_template("emr_prediction.html", error="Model chưa được tải.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
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

# === RUN APP ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
