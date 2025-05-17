import os
import secrets
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import ydata_profiling
import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

app = Flask(__name__)

def combine_model_parts(parts, output):
    with open(output, 'wb') as wfd:
        for part in parts:
            with open(part, 'rb') as fd:
                wfd.write(fd.read())

# Danh sách các phần model đã split
model_parts = [
    'models/best_weights_model.keras.001',
    'models/best_weights_model.keras.002',
    'models/best_weights_model.keras.003',
    'models/best_weights_model.keras.004',
]

# File model gộp lại
combined_model_path = 'models/best_weights_model.keras'

# Nếu file combined chưa tồn tại hoặc bạn muốn luôn gộp lại
if not os.path.exists(combined_model_path):
    print("Đang ghép các phần model lại thành 1 file...")
    combine_model_parts(model_parts, combined_model_path)
    print("Hoàn thành ghép file model.")

# Load model từ file đã ghép
print("Đang load model từ file:", combined_model_path)
model = load_model(combined_model_path)
print("Model đã được load thành công!")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/emr_profile')
def emr_profile():
    return render_template('EMR_Profile.html')

@app.route('/emr_prediction')
def emr_prediction():
    return render_template('EMR_Prediction.html')



@app.route('/emr_profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không có file được tải lên')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_CSV):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                profile = ydata_profiling.ProfileReport(df, minimal=True)
                profile_html = profile.to_html()
                return render_template('emr_profile_result.html', profile_html=profile_html)
            except Exception as e:
                flash(f'Lỗi khi phân tích file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Chỉ cho phép file CSV')
            return redirect(request.url)
    return render_template('emr_profile.html')

@app.route('/emr_prediction', methods=['GET', 'POST'])
def emr_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không có file được tải lên')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_CSV):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                X = df.values  # Tùy chỉnh theo model của bạn
                predictions = model.predict(X)
                pred_labels = predictions.argmax(axis=1) if predictions.shape[1] > 1 else (predictions > 0.5).astype(int)
                df['Prediction'] = pred_labels
                return render_template('emr_prediction_result.html', tables=[df.to_html(classes='table table-striped')], titles=df.columns.values)
            except Exception as e:
                flash(f'Lỗi khi dự đoán: {str(e)}')
                return redirect(request.url)
        else:
            flash('Chỉ cho phép file CSV')
            return redirect(request.url)
    return render_template('emr_prediction.html')

@app.route('/image_analysis', methods=['GET', 'POST'])
def image_analysis():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Không có ảnh được tải lên')
            return redirect(request.url)
        image_file = request.files['image']
        if image_file.filename == '':
            flash('Chưa chọn ảnh')
            return redirect(request.url)
        if image_file and allowed_file(image_file.filename, ALLOWED_EXTENSIONS_IMG):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)[0]
                label = "Nodule" if np.argmax(pred) == 1 else "Non-Nodule"
                confidence = np.max(pred) * 100
                return render_template('image_analysis_result.html', label=label, confidence=confidence, filename=filename)
            except Exception as e:
                flash(f'Lỗi khi phân tích ảnh: {str(e)}')
                return redirect(request.url)
        else:
            flash('Chỉ cho phép file ảnh PNG, JPG, JPEG')
            return redirect(request.url)
    return render_template('image_analysis.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
