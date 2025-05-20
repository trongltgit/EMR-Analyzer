import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Giới hạn upload file lên 32MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_PROFILE_REPORTS = os.path.join(BASE_DIR, 'static', 'profile_reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_PROFILE_REPORTS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

model = None

def merge_model_parts():
    """Ghép các phần .keras.001, .keras.002,... thành file .keras"""
    part_files = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(MODEL_FILENAME + ".")
    ])
    if not part_files:
        print("⚠️ Không thấy các phần model.")
        return False
    print(f"🔧 Ghép model từ các phần: {part_files}")
    try:
        with open(MODEL_PATH, 'wb') as outfile:
            for part in part_files:
                with open(os.path.join(MODELS_DIR, part), 'rb') as pf:
                    while True:
                        chunk = pf.read(1024 * 1024)
                        if not chunk:
                            break
                        outfile.write(chunk)
        print(f"✅ Ghép model thành công! Đã tạo {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

def try_load_model():
    """Tải mô hình với kiểm tra lỗi chi tiết"""
    global model
    try:
        print(f"🔍 Kiểm tra model ở: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print("🔍 File model chưa tồn tại, thử merge...")
            merged = merge_model_parts()
            if not merged:
                print("⚠️ Model chưa được ghép.")
        
        if os.path.exists(MODEL_PATH):
            if not os.access(MODEL_PATH, os.R_OK):
                print(f"⚠️ Không có quyền truy cập vào {MODEL_PATH}. Kiểm tra lại!")
                return
            
            print("🔍 Đang load model...")
            model = load_model(MODEL_PATH)
            print("✅ Model đã được load.")
        else:
            print("⚠️ Không tìm thấy file model sau khi merge.")
            model = None
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        model = None

try_load_model()



# Cài đặt các thư viện cần thiết
!pip install ydata-profiling
!pip install gdown

import pandas as pd
import os
import gdown
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split

# File ID từ Google Drive (thay bằng ID thực tế)
file_id = '1t_VjKggCbXBYwJowLbC898WKsFuxiNG9'  # Thay bằng ID thực tế
output_file = 'healthcare_dataset.csv'

# URL để tải file từ Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

# Kiểm tra nếu file đã được tải thành công
if not os.path.exists(output_file):
    print(f"Error: File not found at {output_file}")
else:
    data = pd.read_csv(output_file)

    # Tạo báo cáo
    profile = ProfileReport(data, title='Pandas Profiling Report Dummy')

    # Xuất báo cáo ra file
    profile.to_file(output_file="report_dummy.html")
    profile.to_file(output_file="report_dummy.json")

    # Tách tập huấn luyện và kiểm thử
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)

    # Hiển thị báo cáo (chỉ trong notebook)

profile






# File ID từ Google Drive (thay bằng ID thực tế)
file_id = '1EpAgsWQSXi7CsUO8mEQDGAJyjdfN0T6n'  # Thay bằng ID thực tế
output_file = 'best_weights_model.keras'

# URL để tải file từ Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

# Kiểm tra nếu file đã được tải thành công
if not os.path.exists(output_file):
    print(f"Error: File not found at {output_file}")
else:
   from tensorflow import keras

model = keras.models.load_model("best_weights_model.keras")







@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route("/check-model")
def check_model():
    """Kiểm tra sự tồn tại của file model"""
    if os.path.exists(MODEL_PATH):
        return "✅ File tồn tại!"
    else:
        return "❌ Không tìm thấy file!"

@app.route('/emr_profile.html', methods=['GET', 'POST'])
def emr_profile():
    """Xử lý tải file và tạo báo cáo hồ sơ EMR"""
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template("emr_profile.html", error="Vui lòng chọn file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                return render_template("emr_profile.html", error="File không hợp lệ (chỉ nhận .csv, .xls, .xlsx).")

            if df.shape[0] > 100_000 or df.shape[1] > 100:
                error = "File quá lớn (hơn 100.000 dòng hoặc 100 cột), không thể sinh báo cáo profile."
                return render_template("emr_profile.html", error=error)

            try:
                profile = ProfileReport(df, title="EMR Report", minimal=True)
                report_path = os.path.join(STATIC_PROFILE_REPORTS, 'report.html')
                profile.to_file(report_path)
                return redirect(url_for('static', filename='profile_reports/report.html'))
            except MemoryError:
                error = "File quá lớn, không thể sinh báo cáo profile (thiếu bộ nhớ RAM server)."
            except Exception as e:
                error = f"Lỗi khi sinh báo cáo: {e}"
        except Exception as e:
            error = f"Lỗi khi phân tích: {e}"

        return render_template("emr_profile.html", error=error)

    return render_template("emr_profile.html", error=error)

@app.route('/emr_prediction.html', methods=['GET', 'POST'])
def emr_prediction():
    """Xử lý dự đoán từ ảnh hồ sơ EMR"""
    prediction = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error = "Vui lòng chọn ảnh."
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        global model
        if model is None:
            try_load_model()
        if model is None:
            error = (
                f"Model chưa được tải hoặc không tồn tại trên server ({MODEL_PATH}).<br>"
                "Hãy kiểm tra lại log server, đảm bảo đã upload đủ các phần .keras.001, .keras.002,... vào thư mục <b>models</b>!"
            )
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0) / 255.0
            pred = model.predict(img_array)
            prediction = "Nodule" if pred[0][0] > 0.5 else "Non-Nodule"
        except Exception as e:
            error = f"Lỗi khi dự đoán: {e}"

    return render_template("emr_prediction.html", prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
