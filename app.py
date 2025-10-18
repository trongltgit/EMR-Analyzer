import os
import io
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask import send_from_directory # Cần thiết cho việc phục vụ file uploads

# --- Cấu hình ứng dụng ---
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_emr_app' 

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB

# --- Hàm tiện ích (Không thay đổi) ---

def allowed_file(filename):
    """Kiểm tra phần mở rộng của file có được phép không."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_emr_data(df):
    """Mô phỏng quá trình phân tích dữ liệu EMR."""
    try:
        if df.empty:
            return "<h3>Báo cáo Phân tích</h3><p>Tập dữ liệu rỗng. Không có dữ liệu để phân tích.</p>"

        num_rows = len(df)
        if 'Diagnosis' not in df.columns:
            df['Diagnosis'] = [f'Bệnh {i % 5 + 1}' for i in range(num_rows)]
        if 'Age' not in df.columns:
            df['Age'] = [25 + (i % 30) for i in range(num_rows)]

        report_html = "<h3>Báo cáo Phân tích Dữ liệu EMR</h3>"
        report_html += f"<p>Tổng số hồ sơ được phân tích: <strong>{num_rows}</strong></p>"

        top_diagnoses = df['Diagnosis'].value_counts().head(5)
        report_html += "<h4>5 Chẩn đoán phổ biến nhất:</h4>"
        report_html += "<table><tr><th>Chẩn đoán</th><th>Số lượng</th></tr>"
        for diagnosis, count in top_diagnoses.items():
            report_html += f"<tr><td>{diagnosis}</td><td>{count}</td></tr>"
        report_html += "</table>"
        
        avg_age = df['Age'].mean()
        report_html += f"<p style='margin-top: 15px;'>Tuổi trung bình của bệnh nhân: <strong>{avg_age:.1f}</strong></p>"
        
        return report_html

    except Exception as e:
        app.logger.error(f"Lỗi khi mô phỏng phân tích: {e}")
        return f"<h3>Lỗi Phân tích</h3><p>Không thể tạo báo cáo: {e}</p>"

def predict_emr_image(image_path):
    """Mô phỏng quá trình dự đoán EMR từ hình ảnh."""
    filename = os.path.basename(image_path)
    if 'nodule' in filename.lower() or 'tumor' in filename.lower():
        result = "Kết quả: **Nodule (U bướu)** - Độ tin cậy: 95.2%"
    else:
        result = "Kết quả: **Non-Nodule (Không phải u bướu)** - Độ tin cậy: 88.7%"
    return result

# --- Routes của ứng dụng ---

@app.route('/', methods=['GET'])
def dashboard():
    """Trang Dashboard chính."""
    error = session.pop('error', None)
    return render_template('dashboard.html', error=error)

@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    """Route 1: Phân tích Hồ sơ Bệnh án EMR (CSV/Excel)"""
    report_url = None
    error = session.pop('error', None)

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = 'Vui lòng chọn một file.'
            return render_template('emr_profile.html', error=error)
        
        file = request.files['file']
        if not allowed_file(file.filename):
            error = 'Định dạng file không hợp lệ. Chỉ chấp nhận .csv, .xlsx, .xls.'
            return render_template('emr_profile.html', error=error)

        try:
            file_content = file.read()
            df = None
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()

            # Đảm bảo xử lý file được tải lên thành công
            if file_extension == 'csv':
                try:
                    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(file_content.decode('latin-1')))
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(file_content))
            
            if df is None or df.empty:
                error = "File dữ liệu rỗng hoặc không đọc được."
                return render_template('emr_profile.html', error=error)
                
            report_url = analyze_emr_data(df)

        except Exception as e:
            app.logger.error(f"Lỗi xử lý file EMR: {e}")
            error = f"Lỗi không xác định khi xử lý file: {e}"

    return render_template('emr_profile.html', report_url=report_url, error=error)

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    """Route 2: Dự đoán EMR Chuyên sâu (Hình ảnh/Model)"""
    prediction = None
    image_path = None
    error = session.pop('error', None)

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = 'Vui lòng chọn một file hình ảnh.'
            return render_template('emr_prediction.html', error=error)
        
        file = request.files['file']

        if not allowed_file(file.filename):
            error = 'Định dạng file không hợp lệ. Chỉ chấp nhận .jpg, .jpeg, .png.'
            return render_template('emr_prediction.html', error=error)
            
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Lưu file vào thư mục UPLOAD_FOLDER
            file.save(filepath)
            
            # Đường dẫn ảnh để hiển thị trong HTML (Sử dụng tên hàm route)
            image_path = url_for('uploaded_file', filename=filename)
            
            prediction = predict_emr_image(filepath)

        except Exception as e:
            app.logger.error(f"Lỗi xử lý file hình ảnh: {e}")
            error = f"Lỗi không xác định khi xử lý hình ảnh: {e}"

    return render_template('emr_prediction.html', prediction=prediction, image_path=image_path, error=error)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Route để phục vụ các file đã upload (hình ảnh)."""
    # Sử dụng send_from_directory để Flask biết cách phục vụ file tĩnh
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/ping_server', methods=['POST'])
def ping_server():
    """Route kiểm tra sức khỏe máy chủ."""
    return {'message': 'Server is running!'}, 200


if __name__ == '__main__':
    app.run(debug=True)
