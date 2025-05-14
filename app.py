from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# Initialize Flask app
app = Flask(__name__)

# Setup folders
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/best_weights_model.keras'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model safely
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print(f"❌ Model file not found at: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('csv_file')
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            df = pd.read_csv(filepath)
            profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
            profile.to_file("templates/EMR_Profile.html")
            return redirect(url_for('view_profile'))
        except Exception as e:
            return f"❌ Failed to generate profile: {e}"
    return "⚠️ Invalid file format. Please upload a CSV file."

@app.route('/view_profile')
def view_profile():
    return render_template('EMR_Profile.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files.get('image_file')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            image = Image.open(filepath).resize((224, 224)).convert('RGB')
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array)
            result = 'Nodule' if prediction[0][0] > 0.5 else 'Non-Nodule'
            return render_template('EMR_Prediction.html', result=result)
        except Exception as e:
            return f"❌ Error processing image: {e}"
    return "⚠️ No image uploaded."

@app.route('/view_prediction')
def view_prediction():
    return render_template('EMR_Prediction.html')

# Required for Render deployment
app_port = int(os.environ.get("PORT", 10000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app_port)
