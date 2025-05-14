from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model parts
model_dir = 'models'
model_file = os.path.join(model_dir, 'best_weights_model.keras')
model = load_model(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files['csv_file']
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
        profile.to_file("templates/EMR_Profile.html")
        return redirect(url_for('view_profile'))
    return "Invalid file format. Please upload a CSV file."

@app.route('/view_profile')
def view_profile():
    return render_template('EMR_Profile.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image_file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = Image.open(filepath).resize((224, 224)).convert('RGB')
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        result = 'Nodule' if prediction[0][0] > 0.5 else 'Non-Nodule'
        return render_template('EMR_Prediction.html', result=result)
    return "No image uploaded."

@app.route('/view_prediction')
def view_prediction():
    return render_template('EMR_Prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
