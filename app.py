import os
import secrets
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# =======================================================
# 🔧 CẤU HÌNH FLASK APP
# =======================================================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 🔐 Sinh tự động mỗi lần chạy
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =======================================================
# 📦 GHÉP FILE MODEL (4 phần nhỏ -> 1 file .keras)
# =======================================================
MODEL_DIR = "models"
MERGED_MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")

MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.keras.{i:03d}")
    for i in range(1, 5)
]

def merge_model_parts():
    """Ghép 4 phần model nếu file chính chưa tồn tại."""
    if os.path.exists(MERGED_MODEL_PATH):
        print("✅ Model đã tồn tại:", MERGED_MODEL_PATH)
        return True

    print("🔄 Đang ghép các phần model...")
    try:
        with open(MERGED_MODEL_PATH, "wb") as outfile:
            for part in MODEL_PARTS:
                if os.path.exists(part):
                    print("🧩 Thêm:", os.path.basename(part))
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())
                else:
                    print(f"⚠️ Thiếu file: {part}")
                    return False
        print("✅ Ghép model thành công!")
        return True
    except Exception as e:
        print("❌ Lỗi khi ghép model:", e)
        return False


# =======================================================
# 🧠 LOAD MODEL KERAS
# =======================================================
model = None
if merge_model_parts():
    try:
        model = load_model(MERGED_MODEL_PATH)
        print("✅ Model đã được load thành công.")
    except Exception as e:
        print("❌ Không thể load model:", e)
else:
    print("⚠️ Model chưa sẵn sàng (thiếu file hoặc lỗi ghép).")


# =======================================================
# 🏠 TRANG CHỦ / DASHBOARD
# =======================================================
@app.route('/')
def dashboard_page():
    # Nếu bạn chưa có dashboard.html, có thể tạo đơn giản:
    # <h1>Dashboard</h1> <a href="/predict">Dự đoán</a> <a href="/profile">Hồ sơ</a>
    return render_template('dashboard.html')


# =======================================================
# 🩺 TRANG DỰ ĐOÁN EMR (ẢNH)
# =======================================================
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    error = None
    prediction = None
    image_path = None

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

                # Xử lý ảnh đầu vào
                img = Image.open(save_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Dự đoán
                preds = model.predict(img_array)
                predicted_class = np.argmax(preds, axis=1)[0]
                confidence = np.max(preds)

                prediction = f"Kết quả dự đoán: <b>Nhóm {predicted_class}</b> (Độ tin cậy: {confidence:.2%})"
                image_path = f"/{save_path.replace(os.sep, '/')}"

            except Exception as e:
                error = f"Lỗi khi xử lý ảnh hoặc dự đoán: {str(e)}"

    return render_template(
        'emr_prediction.html',
        prediction=prediction,
        error=error,
        image_path=image_path
    )


# =======================================================
# 📊 TRANG PHÂN TÍCH HỒ SƠ EMR (CSV/Excel)
# =======================================================
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
                report_html = f"""
                <p><b>Đã tải file:</b> {file.filename}</p>
                <p>Phân tích thành công! (Bản mẫu)</p>
                """
            except Exception as e:
                error = f"Lỗi khi phân tích hồ sơ: {str(e)}"

    return render_template('emr_profile.html', report_url=report_html, error=error)


# =======================================================
# 🚀 CHẠY APP
# =======================================================
if __name__ == '__main__':
    print("\n🌐 Ứng dụng EMR đang khởi động trên http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
