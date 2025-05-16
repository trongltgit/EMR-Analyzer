from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import pandas as pd
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Load model once
model_path = 'models/best_model.keras'
if not os.path.exists(model_path):
    app.logger.error(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # adjust to your model's expected input
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

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'File not found'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Invalid filename'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        profile = ProfileReport(df, title="EMR Profile Report", explorative=True)
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_report.html')
        profile.to_file(profile_path)
        return jsonify({'report_url': f'/uploads/profile_report.html'})
    except Exception as e:
        app.logger.error(f"CSV profiling error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Image not found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Invalid image filename'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction[0][0]
        elif prediction.ndim == 1:
            prediction = prediction[0]
        else:
            raise ValueError("Unexpected prediction output shape")
        result = 'Nodule' if prediction >= 0.5 else 'Non-Nodule'
        return jsonify({'prediction': result})
    except Exception as e:
        app.logger.error(f"Image prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
