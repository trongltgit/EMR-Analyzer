import os
import secrets
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import ydata_profiling
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

# Tái tạo kiến trúc model giống lúc training
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(your_input_shape,)))  # thay your_input_shape
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # hoặc sigmoid tùy bài toán
    return model

# Tạo model
model = create_model()

# Load weights (chỉ cần prefix, không cần phần .001, .002...)
model.load_weights('models/best_weights_model.keras')


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
