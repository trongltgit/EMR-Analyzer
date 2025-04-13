import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import gdown

app = Flask(__name__)
CORS(app)

# === Config ===
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # 👈 Thay bằng ID của bạn
MODEL_FILE_NAME = "best_weights_model.keras"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

# === Tải model từ Google Drive nếu chưa có ===
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("🧠 Model chưa tồn tại, đang tải từ Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Tải model thành công!")

# === Tải model ===
model = None

def load_model():
    global model
    if model is None:
        download_model_if_needed()
        print("📦 Đang tải model vào bộ nhớ...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Mô hình đã được load!")

# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model()

        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh được gửi!'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Tên file rỗng!'}), 400

        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        print("📊 Kết quả dự đoán:", predictions)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        print(f"❌ Lỗi trong /predict: {str(e)}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# === Chạy server (chỉ khi chạy cục bộ) ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
