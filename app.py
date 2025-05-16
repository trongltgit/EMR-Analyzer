import os
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from ydata_profiling import ProfileReport
import py7zr

app = Flask(__name__)
CORS(app)

# === CONFIG ===
MODEL_FOLDER = "models"
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_weights_model.keras")
MODEL_ARCHIVE = os.path.join(MODEL_FOLDER, "best_weights_model.7z")
CHUNK_NAMES = [
    "best_weights_model.keras.001",
    "best_weights_model.keras.002",
    "best_weights_model.keras.003",
    "best_weights_model.keras.004"
]
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

# === JOIN & EXTRACT MODEL ===
def prepare_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already extracted.")
        return

    print("🔧 Reconstructing model from chunks...")
    with open(MODEL_ARCHIVE, 'wb') as archive:
        for chunk in CHUNK_NAMES:
            chunk_path = os.path.join(MODEL_FOLDER, chunk)
            with open(chunk_path, 'rb') as part:
                archive.write(part.read())

    print("📦 Extracting .7z archive...")
    with py7zr.SevenZipFile(MODEL_ARCHIVE, mode='r') as archive:
        archive.extractall(path=MODEL_FOLDER)

    print("✅ Model extracted successfully.")

# === INIT ===
os.makedirs(MODEL_FOLDER, exist_ok=True)
prepare_model()
model = load_model(MODEL_PATH)

# === UTILITY ===
def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

# === ROUTES ===
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Invalid image file extension"}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        preds = model.predict(image_array)
        class_index = int(np.argmax(preds, axis=1)[0])
        class_label = "Nodule" if class_index == 1 else "Non-Nodule"
        confidence = float(np.max(preds))

        return jsonify({"prediction": class_label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/profile_csv', methods=['POST'])
def profile_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No CSV file provided"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
        return jsonify({"error": "Invalid CSV file extension"}), 400

    try:
        df = pd.read_csv(file)
        profile = ProfileReport(df, minimal=True)
        os.makedirs("static", exist_ok=True)
        output_path = "static/profile_report.html"
        profile.to_file(output_path)
        return send_file(output_path)

    except Exception as e:
        return jsonify({"error": f"CSV profiling failed: {str(e)}"}), 500

# === ENTRYPOINT ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
