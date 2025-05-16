import os
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd
from ydata_profiling import ProfileReport

app = Flask(__name__)
CORS(app)

# Load model once at startup
MODEL_PATH = "models/best_weights_model.keras"  # Điều chỉnh đường dẫn đúng
model = load_model(MODEL_PATH)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_CSV_EXTENSIONS = {'csv'}


def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


@app.route('/')
def index():
    # Trang chính, có thể redirect sang dashboard hoặc upload page
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
        image = image.resize((224, 224))  # Hoặc kích thước bạn dùng khi training
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # (1, H, W, C)

        preds = model.predict(image_array)
        # Giả sử model output dạng xác suất 2 lớp
        class_index = np.argmax(preds, axis=1)[0]
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
        output_path = "static/profile_report.html"
        profile.to_file(output_path)
        return send_file(output_path)

    except Exception as e:
        return jsonify({"error": f"CSV profiling failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Chạy app Flask trên 0.0.0.0 port 5000, phù hợp với Render
    app.run(host='0.0.0.0', port=5000, debug=False)
