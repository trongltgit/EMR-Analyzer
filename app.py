import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import io
import tempfile
import uuid
from ydata_profiling import ProfileReport

app = Flask(__name__)

# === Tự động merge model nếu chưa tồn tại ===
MODEL_PATH = 'models/best_weights_model.keras'
if not os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'wb') as output_file:
            for i in range(1, 5):
                part_filename = f'models/best_weights_model.keras.00{i}'
                with open(part_filename, 'rb') as part:
                    output_file.write(part.read())
        print("✅ Model merged successfully.")
    except Exception as e:
        print(f"❌ Model merge failed: {e}")

# === Load model ===
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print("⚠️ Model file not found.")

# === Trang 1: Phân tích EMR hồ sơ bệnh án ===
@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    report_path = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                filename = file.filename
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    error = "Chỉ hỗ trợ định dạng CSV và Excel."
                    return render_template('emr_profile.html', error=error)

                profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
                temp_dir = tempfile.gettempdir()
                unique_filename = f"profile_{uuid.uuid4().hex}.html"
                report_path = os.path.join(temp_dir, unique_filename)
                profile.to_file(report_path)
                return send_file(report_path, as_attachment=True)
            except Exception as e:
                error = f"Lỗi khi xử lý file: {e}"
        else:
            error = "Vui lòng chọn một file để upload."

    return render_template('emr_profile.html', error=error)

# === Trang 2: Dự đoán từ ảnh y tế ===
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    prediction = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                image = Image.open(file).convert('RGB')
                image = image.resize((224, 224))  # tùy thuộc vào input model
                img_array = np.array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                y_pred = model.predict(img_array)[0][0]
                prediction = "Nodule" if y_pred > 0.5 else "Non-Nodule"
            except Exception as e:
                error = f"Lỗi xử lý ảnh: {e}"
        else:
            error = "Vui lòng chọn một ảnh để upload."

    return render_template('emr_prediction.html', prediction=prediction, error=error)




# === Trang chủ điều hướng ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/emr_profile')
def emr_profile():
    return render_template('EMR_Profile.html')

@app.route('/emr_prediction')
def emr_prediction():
    return render_template('EMR_Prediction.html')




if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

