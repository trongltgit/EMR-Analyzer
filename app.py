import os
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import urllib.request
import drive
drive.mount('/content/drive')


model_path = '/content/drive/MyDrive/efficientnet/efficientnet/best_weights_model.keras'
best_model = load_model(model_path)

urllib.request.urlretrieve("https://raw.githubusercontent.com/trongltgit/EMR-Analyzer/refs/heads/main/templates/dashboard.html", "/content/uploader.html")



# Define Flask routes
@app.route("/")
def index():
    return Path('/content/uploader.html').read_text()

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        image_path = '/content/' + file.filename
        file.save(image_path)  # Save the file to a folder named 'uploads'

        # Đọc ảnh và chuyển về kích thước mong muốn (240x240 trong trường hợp này)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (240, 240))
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch

        # Chuẩn hóa dữ liệu (nếu cần)
        # image = image / 255.0

        # Dự đoán nhãn
        prediction = best_model.predict(image)
        binary_prediction = np.round(prediction)

        return json.dumps(binary_prediction.tolist())

    return 'Error uploading file'

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     filename="server.log",
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Flask app initialization
# app = Flask(__name__, template_folder="templates", static_folder="static")
# CORS(app)

# # Google Drive File ID and Model Path
# file_id = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"  # Update this with the correct file ID
# model_path = "./best_weights_model.keras"

# # Download model file from Google Drive if not present
# def download_model():
#     if not os.path.exists(model_path):
#         url = f"https://drive.google.com/uc?id={file_id}"
#         try:
#             logging.info(f"Downloading model file from Google Drive ID: {file_id}")
#             gdown.download(url, model_path, quiet=False)
#             logging.info(f"Model file downloaded successfully to {model_path}")
#         except Exception as e:
#             logging.error(f"Error downloading model file: {e}", exc_info=True)
#             raise RuntimeError("Failed to download the model file. Check permissions or link.")

# # Load the model
# try:
#     download_model()
#     best_model = load_model(model_path)
#     logging.info("Model loaded successfully!")
# except Exception as e:
#     logging.error(f"Error loading model: {e}", exc_info=True)
#     best_model = None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    if best_model is None:
        return jsonify({'error': 'Model is not loaded. Please try again later.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided!'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename!'}), 400

    try:
        img = Image.open(file).convert('RGB').resize((240, 240))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = best_model.predict(img_array)
        binary_prediction = np.round(prediction).tolist()
        return jsonify({'prediction': binary_prediction})
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment variable
    app.run(host='0.0.0.0', port=port)
