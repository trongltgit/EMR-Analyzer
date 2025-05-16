from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'models'
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Join model parts if needed
if not os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'wb') as f_out:
            for i in range(1, 5):
                part_path = os.path.join(MODEL_DIR, f"{MODEL_FILENAME}.00{i}")
                with open(part_path, 'rb') as f_in:
                    f_out.write(f_in.read())
        print("✅ Model parts joined successfully.")
    except Exception as e:
        print(f"❌ Failed to join model parts: {e}")

# Load model
model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Model loading error: {e}")

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
    file = request.files.get('file')
    if file and file.filename.endswith('.csv'):
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        try:
            df = pd.read_csv(path)
            profile = ProfileReport(df, title="EMR Profiling Report", explorative=True)
            profile_path = os.path.join("templates", "EMR_Profile.html")
            profile.to_file(profile_path)
            return jsonify({"report_url": "/view_profile"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file"}), 400

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        try:
            img = Image.open(path).resize((224, 224)).convert('RGB')
            x = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = model.predict(x)[0][0]
            result = "Nodule" if pred > 0.5 else "Non-Nodule"
            return jsonify({"prediction": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No image uploaded"}), 400

@app.route('/view_profile')
def view_profile():
    return render_template('EMR_Profile.html')

@app.route('/view_prediction')
def view_prediction():
    return render_template('EMR_Prediction.html')

app_port = int(os.environ.get("PORT", 10000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app_port)
