# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cấu hình Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx', 'png', 'jpg', 'jpeg'}

# Đảm bảo thư mục upload tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Đường dẫn và Tên model
MODEL_BASE_PATH = 'models/best_weights_model.keras'
MODEL_PARTS_PATTERN = MODEL_BASE_PATH + '.%03d' # .001, .002, ...
GLOBAL_MODEL = None
GRAPH = None # Khởi tạo Graph cho TensorFlow để đảm bảo thread-safe

# --- Hàm hỗ trợ ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def merge_model_parts():
    """Ghép các phần của model (nếu tồn tại) thành một file duy nhất."""
    merged_path = MODEL_BASE_PATH
    if os.path.exists(merged_path):
        return True # File đã tồn tại, không cần ghép
        
    part_paths = []
    i = 1
    while True:
        part_path = MODEL_PARTS_PATTERN % i
        if not os.path.exists(part_path):
            break
        part_paths.append(part_path)
        i += 1
    
    if len(part_paths) < 1:
        # Không tìm thấy bất kỳ phần nào của model
        return False
        
    try:
        with open(merged_path, 'wb') as outfile:
            for part_file in part_paths:
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        return True
    except Exception as e:
        print(f"Lỗi khi ghép các phần model: {e}")
        # Xóa file đã ghép nếu quá trình ghép lỗi
        if os.path.exists(merged_path):
            os.remove(merged_path)
        return False

def load_ai_model():
    """Tải model AI vào bộ nhớ."""
    global GLOBAL_MODEL, GRAPH
    if GLOBAL_MODEL:
        print("Model đã được tải sẵn.")
        return True
    
    if not merge_model_parts():
        print("LỖI: Không tìm thấy các phần model để ghép hoặc file model gốc.")
        return False

    try:
        # Sử dụng tf.compat.v1.get_default_graph() để xử lý vấn đề thread-safe trong Flask
        # đối với các phiên bản TensorFlow cũ
        GRAPH = tf.compat.v1.get_default_graph() 
        with GRAPH.as_default():
            GLOBAL_MODEL = load_model(MODEL_BASE_PATH)
            print("Model AI đã tải thành công.")
            return True
    except Exception as e:
        print(f"LỖI TẢI MODEL: {e}")
        return False

# Gọi hàm tải model khi ứng dụng khởi động lần đầu
with app.app_context():
    load_ai_model()

# --- Định tuyến (Routes) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Trang Đăng nhập (hoặc Trang Chủ)"""
    if request.method == 'POST':
        # Logic đăng nhập đơn giản: chỉ cần POST là chuyển hướng
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard chính"""
    return render_template('dashboard.html')

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    """Phân tích hồ sơ EMR (Data Analysis)"""
    error = None
    report_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Vui lòng chọn file dữ liệu."
            return render_template('emr_profile.html', error=error)
        
        file = request.files['file']
        
        if file.filename == '':
            error = "Vui lòng chọn file dữ liệu."
            return render_template('emr_profile.html', error=error)
            
        if file and allowed_file(file.filename):
            try:
                # Đọc file vào bộ nhớ
                file_content = file.read()
                if not file_content:
                    error = "File dữ liệu rỗng. Vui lòng kiểm tra lại file CSV/Excel."
                    return render_template('emr_profile.html', error=error)
                    
                # Sử dụng io.BytesIO để đọc nội dung file từ bộ nhớ
                file_stream = io.BytesIO(file_content)
                
                filename = secure_filename(file.filename)
                
                # Xác định loại file (CSV hay Excel)
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_stream)
                else: # Excel (xls, xlsx)
                    df = pd.read_excel(file_stream)
                    
                # KIỂM TRA DỮ LIỆU
                if df.empty or df.shape[1] == 0:
                     error = "Lỗi đọc file: File dữ liệu rỗng hoặc không có cột để phân tích."
                     return render_template('emr_profile.html', error=error)
                     
                # Thực hiện phân tích: Ví dụ đơn giản là hiển thị 5 dòng đầu và thống kê
                analysis_result = "Phân tích thành công:<br>"
                analysis_result += "<h3>5 dòng dữ liệu đầu tiên:</h3>"
                analysis_result += df.head().to_html(classes='table table-striped')
                analysis_result += "<h3>Thống kê tổng quan:</h3>"
                analysis_result += df.describe().to_html(classes='table table-striped')
                
                report_url = analysis_result # Dùng report_url để truyền HTML đã phân tích
                
            except pd.errors.EmptyDataError:
                error = "Lỗi đọc file: File dữ liệu rỗng hoặc không đúng định dạng CSV/Excel."
            except Exception as e:
                error = f"Lỗi khi đọc file dữ liệu: {e}"
                
    return render_template('emr_profile.html', error=error, report_url=report_url)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    """Phân tích EMR Chuyên sâu (Image Prediction)"""
    error = None
    prediction = None
    image_path = None
    
    if not GLOBAL_MODEL:
        error = "Model chưa được tải hoặc không tồn tại trên server. Hãy kiểm tra lại log server và đảm bảo đã upload đủ các phần file model vào thư mục models!"
        return render_template('emr_prediction.html', error=error)

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Vui lòng chọn file ảnh."
            return render_template('emr_prediction.html', error=error)
        
        file = request.files['file']
        
        if file.filename == '':
            error = "Vui lòng chọn file ảnh."
            return render_template('emr_prediction.html', error=error)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(upload_path)
                image_path = url_for('uploaded_file', filename=filename)

                # PHÂN TÍCH ẢNH DÙNG MODEL
                img = image.load_img(upload_path, target_size=(224, 224)) # Kích thước target tùy thuộc model
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0 # Chuẩn hóa
                
                # Thực hiện dự đoán
                with GRAPH.as_default(): # Dùng graph đã định nghĩa
                    predictions = GLOBAL_MODEL.predict(img_array)

                # Xử lý kết quả dự đoán (Ví dụ cho phân loại nhị phân)
                if predictions[0] > 0.5:
                    prediction = f"Kết quả dự đoán: Nodule (U/Nốt sần) với độ tin cậy {predictions[0][0]*100:.2f}%"
                else:
                    prediction = f"Kết quả dự đoán: Non-Nodule (Không phải U/Nốt sần) với độ tin cậy {(1-predictions[0][0])*100:.2f}%"
                
            except Exception as e:
                error = f"Lỗi trong quá trình xử lý hoặc dự đoán: {e}"
        else:
            error = "Định dạng file không được hỗ trợ. Vui lòng chọn ảnh (jpg, png)."
                
    return render_template('emr_prediction.html', error=error, prediction=prediction, image_path=image_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Phục vụ file đã upload"""
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    # Đảm bảo thư mục static/uploads tồn tại nếu chạy cục bộ
    static_upload_dir = os.path.join('static', app.config['UPLOAD_FOLDER'])
    if not os.path.exists(static_upload_dir):
        os.makedirs(static_upload_dir)
        
    app.run(debug=True)
