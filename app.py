import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# ========================
# ⚙️ Cấu hình Flask
# ========================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ========================
# 📦 Nạp model Keras (tự động ghép file .001-.004)
# ========================
MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_weights_model.keras')

if not os.path.exists(MODEL_PATH):
    # Ghép các phần model nếu chưa có file hoàn chỉnh
    parts = [f for f in sorted(os.listdir(MODEL_FOLDER)) if f.startswith('best_weights_model.keras.')]
    if parts:
        with open(MODEL_PATH, 'wb') as full_model:
            for part in parts:
                with open(os.path.join(MODEL_FOLDER, part), 'rb') as p:
                    full_model.write(p.read())

# Tải model (nếu có)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model Keras đã được nạp thành công!")
    except Exception as e:
        print("⚠️ Lỗi khi nạp model:", e)
else:
    print("⚠️ Không tìm thấy model tại:", MODEL_PATH)

# ========================
# 🔐 Route: Trang chủ (index.html)
# ========================
@app.route('/')
def home():
    return render_template('index.html')

# ========================
# 🔐 Route: Dashboard (chỉ vào được sau khi login)
# ========================
@app.route('/dashboard')
def dashboard():
    # Kiểm tra session đăng nhập
    if not session.get('logged_in'):
        return redirect(url_for('home'))
    return render_template('dashboard.html')

# ========================
# 🔐 Route: Xử lý đăng nhập
# ========================
@app.route('/login', methods=['POST'])
def login():
    # Lấy dữ liệu từ form HTML (dù script login dùng client-side, route này vẫn có thể xử lý POST)
    user = request.form.get('userID')
    password = request.form.get('password')

    if user == 'user_demo' and password == 'Test@123456':
        session['logged_in'] = True
        session['user'] = user
        return redirect(url_for('dashboard'))
    else:
        return render_template('index.html', error="Sai ID hoặc mật khẩu!")

# ========================
# 🚪 Đăng xuất
# ========================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# ========================
# 🧠 Trang phân tích hồ sơ EMR (text-based)
# ========================
@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if not session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        # Giả sử ở đây bạn upload hồ sơ CSV / JSON / TXT
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            return render_template('emr_profile.html', result=f"Tải lên thành công: {filename}")

    return render_template('emr_profile.html', result=None)

# ========================
# 🩻 Trang phân tích ảnh y tế (dùng model Keras)
# ========================
@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if not session.get('logged_in'):
        return redirect(url_for('home'))

    result = None
    if request.method == 'POST' and 'image' in request.files:
        img_file = request.files['image']
        filename = secure_filename(img_file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(save_path)

        if model:
            try:
                img = image.load_img(save_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                pred = model.predict(img_array)
                result = f"Kết quả dự đoán: {np.argmax(pred)}"
            except Exception as e:
                result = f"Lỗi xử lý ảnh: {e}"
        else:
            result = "⚠️ Model chưa được nạp!"

    return render_template('emr_prediction.html', result=result)

# ========================
# 📁 Route truy cập file upload
# ========================
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ========================
# 🚀 Chạy app
# ========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
