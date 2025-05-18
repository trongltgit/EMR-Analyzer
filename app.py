import os
from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import uuid
from ydata_profiling import ProfileReport

app = Flask(__name__)

# === Cấu hình ===
MODEL_DIR = 'models'
MERGED_MODEL_PATH = os.path.join(MODEL_DIR, 'best_weights_model.keras')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Ghép model từ các phần nếu cần ===
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

# === Load model ===
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

@app.route('/emr_profile', methods=['GET', 'POST'])
@app.route('/emr_profile.html', methods=['GET', 'POST'])
def emr_profile():
    error = None
    report_url = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                filename = file.filename.lower()
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    error = "Chỉ hỗ trợ file CSV hoặc Excel."
                    return render_template('emr_profile.html', error=error)

                profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
                temp_dir = os.path.join('static', 'reports')
                os.makedirs(temp_dir, exist_ok=True)

                report_filename = f"profile_{uuid.uuid4().hex}.html"
                report_path = os.path.join(temp_dir, report_filename)
                profile.to_file(report_path)

                report_url = '/' + report_path.replace('\\', '/')

            except Exception as e:
                error = f"Lỗi khi xử lý file: {e}"
        else:
            error = "Vui lòng chọn một file để upload."

    return render_template('emr_profile.html', error=error, report_url=report_url)

@app.route('/emr_prediction', methods=['GET', 'POST'])
@app.route('/emr_prediction.html', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    error = None
    image_url = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                image = Image.open(filepath).convert('RGB')
                image = image.resize((224, 224))
                img_array = np.array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                if model:
                    y_pred = model.predict(img_array)[0][0]
                    prediction = "Nodule" if y_pred > 0.5 else "Non-Nodule"
                    image_url = '/' + filepath.replace('\\', '/')
                else:
                    error = "Model chưa được load."

            except Exception as e:
                error = f"Lỗi xử lý ảnh: {e}"
        else:
            error = "Vui lòng chọn một ảnh để upload."

    return render_template('emr_prediction.html', prediction=prediction, error=error, image_url=image_url)

# === RUN APP ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
