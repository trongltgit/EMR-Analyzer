from flask import Flask, render_template, request, jsonify
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

# Load model duy nhất 1 lần
model = tf.keras.models.load_model('models/best_model.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET'])
def emr_profile():
    return render_template('EMR_Profile.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file trong yêu cầu.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        profile = ProfileReport(df, title='EMR Profile Report', explorative=True)
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_report.html')
        profile.to_file(report_path)
        return jsonify({'report_url': f'/uploads/profile_report.html'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/emr_prediction', methods=['GET'])
def emr_prediction():
    return render_template('EMR_Prediction.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh trong yêu cầu.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn ảnh.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]
        label = 'Nodule' if prediction >= 0.5 else 'Non-Nodule'
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
