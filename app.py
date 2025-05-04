import os
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="server.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Allow CORS for all origins

# Google Drive File ID and Model Path
file_id = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
model_path = "./best_weights_model.keras"

# Download the model file from Google Drive if it doesn't exist
def download_model():
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            logging.info(f"Downloading model file from Google Drive ID: {file_id}")
            gdown.download(url, model_path, quiet=False)
            logging.info(f"Model file downloaded successfully to {model_path}")
        except Exception as e:
            logging.error(f"Error downloading model file: {e}", exc_info=True)
            raise RuntimeError("Failed to download the model file. Check permissions or link.")

# Load the model
try:
    download_model()
    best_model = load_model(model_path)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)
    best_model = None

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
