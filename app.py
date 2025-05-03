import os
import shutil
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model as keras_load_model
from flask_cors import CORS
from PIL import Image
import gdown
import logging
from retrying import retry
import py7zr

# Cấu hình logging (ghi log vào file server.log)
logging.basicConfig(level=logging.DEBUG, filename="server.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)  # Cho phép tất cả origin

# Cấu hình thông số cho model
MODEL_FILE_ID = "1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n"
MODEL_FILE_NAME = "best_weights_model.keras"
# Lưu ý: đảm bảo đường dẫn folder này tồn tại trên Render
MODEL_DIR = "./MyDrive/efficientnet/efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
model = None

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_model_gdown(url, output):
    gdown.download(url, output, quiet=False)

def assemble_model():
    """
    Ghép các file .7z đã chia nhỏ và giải nén để lấy file model gốc.
    """
    global model
    try:
        segments = [os.path.join(MODEL_DIR, f"best_weights_model.7z.00{i}") for i in range(1, 5)]
        logging.info("Checking existence of split files: %s", segments)
        for seg in segments:
            if not os.path.exists(seg):
                logging.error("File not found: %s", seg)
                return
        assembled_archive = os.path.join(MODEL_DIR, "best_weights_model.7z")
        with open(assembled_archive, "wb") as outfile:
            for seg in segments:
                with open(seg, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        logging.info("Assembled archive successfully. Extracting...")
        with py7zr.SevenZipFile(assembled_archive, mode="r") as archive:
            archive.extractall(path=MODEL_DIR)
        extracted_model = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
        model = keras_load_model(extracted_model)
        logging.info("Model loaded from extracted file.")
    except Exception as e:
        logging.error("Error assembling or loading model: %s", str(e))
        model = None

def download_model():
    """
    Tải model từ Google Drive nếu chưa có. Nếu thất bại, chuyển sang ghép file .7z.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        try:
            logging.info("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            download_model_gdown(url, MODEL_PATH)
            logging.info("Model downloaded successfully from Drive.")
        except Exception as e:
            logging.warning("Failed to download model from Drive: %s", e)
            assemble_model()

def initialize_model():
    """
    Load model vào bộ nhớ.
    Nếu model chưa được load, thử tải về hoặc ghép file.
    """
    global model
    if model is None:
        download_model()
        if model is None and os.path.isfile(MODEL_PATH):
            try:
                logging.info("Loading model from local file...")
                model = keras_load_model(MODEL_PATH)
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error("Error loading local model: %s", e)

# Thêm endpoint ping để kiểm tra sự phản hồi của máy chủ
@app.route("/ping")
def ping():
    return jsonify({"status": "OK"})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        if model is None:
            initialize_model()
            if model is None:
                logging.error("Model not loaded")
                return jsonify({"error": "Model not loaded"}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image file in request!"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Invalid image file name!"}), 400

        # Xử lý ảnh: chuyển sang RGB và resize về 224x224
        img = Image.open(file).convert("RGB").resize((224,224))
        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        img.close()
        preds = model.predict(x)[0][0]
        cls = "Nodule" if preds > 0.5 else "Non-Nodule"
        return jsonify({"classification": cls, "score": float(preds)})
    except Exception as e:
        logging.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# Error handlers để luôn trả về JSON
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
