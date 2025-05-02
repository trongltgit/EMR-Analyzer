import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import gdown

app = Flask(__name__)
CORS(app)

# Đường dẫn model và URL Google Drive
MODEL_PATH = 'models/best_weights_model.keras'
MODEL_DIR = 'models'
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'  # thay đúng ID nếu cần

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

# Tải model nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("🔽 Model chưa có, đang tải từ Google Drive...")
    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Tải model hoàn tất.")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tên file rỗng'}), 400

    try:
        # Xử lý ảnh
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        predictions = model.predict(img_array)
        confidence = float(predictions[0][0])
        label = 'Nodule' if confidence > 0.5 else 'Non-Nodule'
        return jsonify({'result': label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_file('dashboard.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
