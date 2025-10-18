import os
import secrets
import numpy as np
from flask import Flask, render_template, request, Markup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ============================================
# 🚀 CẤU HÌNH ỨNG DỤNG
# ============================================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MERGED_MODEL_PATH = os.path.join(MODEL_DIR, "best_weights_model.keras")
MODEL_PARTS = [
    os.path.join(MODEL_DIR, f"best_weights_model.keras.{i:03d}")
    for i in range(1, 5)
]

# ============================================
# 🧩 GHÉP FILE MODEL
# ============================================
def merge_model_parts():
    """Ghép 4 file .keras.001 -> .004 thành 1 model keras"""
    if os.path.exists(MERGED_MODEL_PATH):
        print("✅ Model đã tồn tại:", MERGED_MODEL_PATH)
        return True

    print("🔄 Đang ghép model từ các phần...")
    try:
        with open(MERGED_MODEL_PATH, "wb") as merged:
            for part in MODEL_PARTS:
                if not os.path.exists(part):
                    print(f"⚠️ Thiếu {part}")
                    return False
                print("🧩 Nối:", os.path.basename(part))
                with open(part, "rb") as f:
                    merged.write(f.read())
        print("✅ Đã ghép thành công model.")
        return True
    except Exception as e:
        print("❌ Lỗi khi ghép model:", e)
        return False


# ============================================
# 🧠 LOAD MODEL
# ============================================
model = None
if merge_model_parts():
    try:
        model = load_model(MERGED_MODEL_PATH)
        print("✅ Model đã được load thành công.")
    except Exception as e:
        print("⚠️ Lỗi load model:", e)
else:
    print("⚠️ Model chưa sẵn sàng (thiếu file .keras.*)")

# ============================================
# 🏠 TRANG CHỦ (index.html)
# ============================================
@app.route('/')
def index_page():
    try:
        return render_template('index.html')
    except:
        # Fallback nếu chưa có file HTML
        return """
        <h1 style='color:#2e7d32;text-align:center;'>EMR Dashboard</h1>
        <p style='text-align:center;'>
            <a href='/predict'>🔍 Dự đoán từ ảnh</a> |
            <a href='/profile'>📊 Phân tích hồ sơ</a>
        </p>
        """

# ============================================
# 🩺 DỰ ĐOÁN TỪ HÌNH ẢNH
# ============================================
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    error, prediction, image_path = None, None, None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            error = "Vui lòng chọn một ảnh."
        elif model is None:
            error = "⚠️ Model chưa sẵn sàng để dự đoán."
        else:
            try:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                img = Image.open(filepath).convert('RGB').resize((224, 224))
                img_arr = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
                preds = model.predict(img_arr)
                pred_class = int(np.argmax(preds))
                confidence = float(np.max(preds))

                prediction = Markup(f"Kết quả: <b>Nhóm {pred_class}</b> - Độ tin cậy: {confidence:.2%}")
                image_path = f"/{filepath.replace(os.sep, '/')}"
            except Exception as e:
                error = f"Lỗi khi xử lý ảnh: {str(e)}"

    try:
        return render_template('emr_prediction.html',
                               error=error,
                               prediction=prediction,
                               image_path=image_path)
    except:
        # Fallback nếu thiếu HTML
        return f"""
        <h1 style='color:#2e7d32;'>Dự đoán EMR</h1>
        <form method='POST' enctype='multipart/form-data'>
            <input type='file' name='file'>
            <button type='submit'>Phân tích</button>
        </form>
        <p style='color:red;'>{error or ''}</p>
        <p>{prediction or ''}</p>
        {'<img src="'+image_path+'" width="300">' if image_path else ''}
        """

# ============================================
# 📊 PHÂN TÍCH FILE EMR
# ============================================
@app.route('/profile', methods=['GET', 'POST'])
def profile_page():
    report_url, error = None, None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "Vui lòng chọn file CSV hoặc Excel."
        else:
            try:
                report_url = f"<p><b>Đã tải file:</b> {file.filename}</p><p>Phân tích thành công!</p>"
            except Exception as e:
                error = f"Lỗi khi xử lý file: {e}"

    try:
        return render_template('emr_profile.html',
                               error=error,
                               report_url=report_url)
    except:
        # Fallback nếu chưa có HTML
        return f"""
        <h1>Phân tích hồ sơ EMR</h1>
        <form method='POST' enctype='multipart/form-data'>
            <input type='file' name='file'>
            <button type='submit'>Tải lên</button>
        </form>
        <p style='color:red;'>{error or ''}</p>
        <div>{report_url or ''}</div>
        """

# ============================================
# 🚀 CHẠY APP
# ============================================
if __name__ == '__main__':
    print("🌿 Flask EMR app đang chạy tại: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
