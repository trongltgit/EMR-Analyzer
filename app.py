from flask import Flask, render_template, request
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

# Load model chỉ một lần duy nhất
model = tf.keras.models.load_model('models/best_model.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # tùy thuộc vào input shape của model
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    profile_html = None
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df = pd.read_csv(file_path)
            profile = ProfileReport(df, title="EMR Profile Report", explorative=True)
            profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_report.html')
            profile.to_file(profile_path)

            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_html = f.read()

    return render_template('EMR_Profile.html', profile_html=profile_html)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    result = None
    if request.method == 'POST':
        file = request.files['image_file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = preprocess_image(file_path)
            prediction = model.predict(image)[0][0]
            result = 'Nodule' if prediction >= 0.5 else 'Non-Nodule'

    return render_template('EMR_Prediction.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
