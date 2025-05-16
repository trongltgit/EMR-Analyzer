from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import pandas as pd
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model một lần duy nhất
model = tf.keras.models.load_model('models/best_model.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # hoặc thay đổi tùy thuộc input shape
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
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

# API xử lý CSV
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tên file không hợp lệ'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        profile = ProfileReport(df, title="EMR Profile Report", explorative=True)
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_report.html')
        profile.to_file(profile_path)

        # Trả URL để JS frontend mở trong tab mới
        return jsonify({'report_url': f'/uploads/profile_report.html'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API xử lý ảnh
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tên ảnh không hợp lệ'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]
        result = 'Nodule' if prediction >= 0.5 else 'Non-Nodule'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cho phép truy cập file tĩnh trong folder uploads (ví dụ: profile_report.html)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
