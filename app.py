import os
import io
import secrets
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==========================
# 🔧 CẤU HÌNH FLASK APP
# ==========================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 🔐 Tự sinh key bảo mật mỗi lần chạy
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================
# 📦 GHÉP CÁC FILE MODEL .001 -> .004
# ==========================
MODEL_DIR = "models"
MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.keras.{i:03d}") for i in range(1, 5)
]
MERGED_MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

if not os.path.exists(MERGED_MODEL_PATH):
    try:
        print("🔄 Đang ghép các phần model...")
        with open(MERGED_MODEL_PATH, "wb") as f_out:
            for part_path in MODEL_PARTS:
                if os.path.exists(part_path):
                    with open(part_path, "rb") as f_in:
                        f_out.write(f_in.read())
                else:
                    print(f"⚠️ Thiếu file model phần: {part_path}")
        print("✅ Ghép model thành công:", MERGED_MODEL_PATH)
    except Exception as e:
        print("❌ Lỗi khi ghép model:", e)

# ==========================
# 📥 LOAD MODEL KERAS
# ==========================
model = None
try:
    if os.path.exists(MERGED_MODEL_PATH):
        model = load_model(MERGED_MODEL_PATH)
        print("✅ Model đã được load thành công.")
    else:
        print("⚠️ Model chưa sẵn sàng.")
except Exception as e:
    print("❌ Không thể load model:", e)


# ==========================
# 📌 TRANG CHỦ / DASHBOARD
# ==========================
@app.route('/')
def dashboard_page():
    return render_template('dashboard.html')


# ==========================
# 🧠 TRANG DỰ ĐOÁN EMR
# ==========================
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    global model
    prediction = None
    image_path = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            error = "Vui lòng chọn một file ảnh hợp lệ."
        elif model is None:
            error = "⚠️ Model chưa sẵn sàng để dự đoán."
        else:
            try:
                # Lưu file upload
                filename = file.filename
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                # Đọc ảnh
                img = Image.open(save_path).convert('RGB')
                img = img.resize((224, 224))  # kích thước đầu vào của model
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Dự đoán
                preds = model.predict(img_array)
                predicted_class = np.argmax(preds, axis=1)[0]

                prediction = f"Kết quả dự đoán: <b>Nhóm {predicted_class}</b>"
                image_path = f"/{save_path}"

            except Exception as e:
                error = f"Lỗi khi xử lý ảnh hoặc dự đoán: {str(e)}"

    return render_template('emr_prediction.html', prediction=prediction, error=error, image_path=image_path)


# ==========================
# 📊 TRANG PHÂN TÍCH HỒ SƠ EMR
# ==========================
@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    report_html = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "Vui lòng chọn file hồ sơ hợp lệ (.csv, .xlsx, .xls)."
        else:
            try:
                # Giả lập: phân tích file CSV hoặc Excel
                report_html = f"<p><b>Đã tải file:</b> {file.filename}</p><p>Phân tích thành công!</p>"
            except Exception as e:
                error = f"Lỗi khi phân tích hồ sơ: {str(e)}"

    return render_template('emr_profile.html', report_url=report_html, error=error)


# ==========================
# 🚀 CHẠY APP
# ==========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
