import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
from ydata_profiling import ProfileReport 
from werkzeug.utils import secure_filename
import tensorflow as tf
from flask import send_from_directory # Sử dụng để phục vụ file từ thư mục uploads 
import warnings

# Bỏ qua cảnh báo RuntimeWarning từ Pandas
warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__)
# Tăng giới hạn upload file lên 32MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

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

# Định nghĩa các đuôi file ảnh được cho phép
ALLOWED_PREDICTION_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Định nghĩa các đuôi file dữ liệu được cho phép (cho EMR Profile)
ALLOWED_PROFILE_EXTENSIONS = {'csv', 'xls', 'xlsx'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

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
    """Tải model Keras"""
    global model
    try:
        print(f"🔍 Kiểm tra model ở: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print("🔍 File model chưa tồn tại, thử merge...")
            merged = merge_model_parts()
            if not merged:
                print("⚠️ Model chưa được ghép.")
        
        if os.path.exists(MODEL_PATH):
            print("🔍 Đang load model...")
            # Thiết lập biến môi trường để giải quyết vấn đề tương thích TF
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
            # Sử dụng tf.compat.v1.get_default_graph().as_default() để đảm bảo threading hoạt động đúng trong môi trường Flask
            with tf.compat.v1.get_default_graph().as_default():
                model = load_model(MODEL_PATH)
            print("✅ Model đã được load.")
        else:
            print("⚠️ Không tìm thấy file model sau khi merge.")
            model = None
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        model = None

try_load_model()

# Endpoint phục vụ ảnh đã upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile.html', methods=['GET', 'POST'])
def emr_profile():
    """Phân tích hồ sơ EMR và tạo báo cáo ProfileReport."""
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template("emr_profile.html", error="Vui lòng chọn file.")

        filename = secure_filename(file.filename)
        
        if not allowed_file(filename, ALLOWED_PROFILE_EXTENSIONS):
            return render_template("emr_profile.html", error="File không hợp lệ (chỉ nhận .csv, .xls, .xlsx).")
            
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            if filename.lower().endswith('.csv'):
                # Thử đọc CSV, encoding utf-8 là phổ biến nhất
                df = pd.read_csv(filepath)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)

            # 🚀 SỬA LỖI: Xử lý file rỗng hoặc không cột
            if df.empty or df.shape[1] == 0:
                error = "File dữ liệu rỗng, không có cột hoặc dữ liệu hợp lệ. Vui lòng kiểm tra lại file CSV/Excel."
                return render_template("emr_profile.html", error=error)

            try:
                # Tạo báo cáo profile
                profile = ProfileReport(df, title=f"Báo cáo EMR: {filename}", minimal=True)
                report_path = os.path.join(STATIC_PROFILE_REPORTS, 'report.html')
                profile.to_file(report_path)
                return redirect(url_for('static', filename='profile_reports/report.html'))
            except MemoryError:
                error = "File quá lớn, không thể sinh báo cáo profile (Memory Error). Vui lòng thử file nhỏ hơn."
            except Exception as e:
                error = f"Lỗi khi sinh báo cáo Profile: {e}"
        
        except pd.errors.EmptyDataError:
            error = "Lỗi đọc file: File dữ liệu rỗng hoặc không đúng định dạng CSV/Excel."
        except Exception as e:
            error = f"Lỗi khi đọc file dữ liệu: {e}"

        return render_template("emr_profile.html", error=error)

    return render_template("emr_profile.html", error=error)

@app.route('/emr_prediction.html', methods=['GET', 'POST'])
def emr_prediction():
    """Dự đoán hình ảnh EMR bằng model."""
    prediction = None
    error = None
    image_path = None 

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = "Vui lòng chọn ảnh."
            return render_template("emr_prediction.html", prediction=prediction, error=error)
            
        filename = secure_filename(file.filename)

        if not allowed_file(filename, ALLOWED_PREDICTION_EXTENSIONS):
            error = "File không hợp lệ. Vui lòng chọn ảnh định dạng PNG, JPG, hoặc JPEG."
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        global model
        if model is None:
            # Nếu model là None, thử load lại
            try_load_model() 
        
        # Kiểm tra lại model sau khi load
        if model is None:
            error = (
                f"Model chưa được tải hoặc không tồn tại trên server ({MODEL_PATH}). "
                "Hãy kiểm tra lại log server và đảm bảo đã upload đủ các phần model vào thư mục models!"
            )
            return render_template("emr_prediction.html", prediction=prediction, error=error)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        image_path = url_for('uploaded_file', filename=filename) 

        try:
            # Tải ảnh và tiền xử lý
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0) / 255.0
            
            # Thực hiện dự đoán
            with tf.compat.v1.get_default_graph().as_default():
                pred = model.predict(img_array)
                # Giả định đây là model phân loại nhị phân
                prediction = "Nodule" if pred[0][0] > 0.5 else "Non-Nodule"
        except Exception as e:
            error = f"Lỗi khi dự đoán: {e}. Vui lòng kiểm tra định dạng và nội dung ảnh."

    return render_template("emr_prediction.html", prediction=prediction, error=error, image_path=image_path)

if __name__ == '__main__':
    # Chạy ứng dụng Flask
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
