import os
import json
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown
from tensorflow.keras.models import load_model

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="server.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Khởi tạo Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Cho phép mọi nguồn gốc truy cập

# Đường dẫn tải xuống file `best_weights_model.keras`
file_id = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
model_path = "./best_weights_model.keras"

# Tải file từ Google Drive nếu chưa tồn tại
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading model file from Google Drive ID: {file_id}")
    gdown.download(url, model_path, quiet=False)
    logging.info(f"Model file downloaded successfully to {model_path}")

# Load mô hình từ file đã tải xuống
try:
    best_model = load_model(model_path)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)
    raise

# Route home
@app.route('/')
def home():
    return render_template('index.html')

# Route dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Route kiểm tra trạng thái server
@app.route('/ping', methods=['GET'])
def ping():
    try:
        return jsonify({'message': 'Server is running!'}), 200
    except Exception as e:
        logging.error(f"Error in /ping: {e}", exc_info=True)
        return jsonify({'error': 'Server Ping Failed!'}), 500

# Route kiểm tra trạng thái mô hình
@app.route('/model-status', methods=['GET'])
def model_status():
    try:
        if best_model is not None:
            return jsonify({'status': 'Model is loaded successfully'}), 200
        else:
            return jsonify({'status': 'Model is not loaded'}), 503
    except Exception as e:
        logging.error(f"Error checking model status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Route dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is present in the request
        if 'image' not in request.files:
            logging.error("No image file found in the request.")
            return jsonify({'error': 'No image file provided!'}), 400

        file = request.files['image']
        if file.filename == '':
            logging.error("Empty filename provided in the request.")
            return jsonify({'error': 'Empty filename!'}), 400

        # Process the image
        logging.info("Processing image for prediction...")
        try:
            img = Image.open(file).convert('RGB').resize((240, 240))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            img.close()
        except Exception as e:
            logging.error(f"Error processing image: {e}", exc_info=True)
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        # Perform prediction
        logging.info("Making prediction...")
        try:
            preds = best_model.predict(img_array)
            binary_prediction = np.round(preds).astype(int).tolist()
            logging.info(f"Prediction successful: {binary_prediction}")
            return jsonify({'prediction': binary_prediction})
        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in /predict: {e}", exc_info=True)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Chạy ứng dụng
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Cổng mặc định là 5000 nếu biến PORT không tồn tại
    app.run(host='0.0.0.0', port=port)
