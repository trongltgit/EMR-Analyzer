import os
import uuid
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from ydata_profiling import ProfileReport
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROFILE_FOLDER'] = 'static/profile'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)

# Load model
model_path = 'models/best_weights_model.keras'
model = load_model(model_path)

def allowed_file(filename, types=None):
    if not types:
        types = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types


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



@app.route('/emr-profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename, {'csv'}):
            filename = str(uuid.uuid4()) + '.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            profile = ProfileReport(df, title="EMR Data Profile", explorative=True)
            profile_name = f"profile_{uuid.uuid4().hex}.html"
            profile_path = os.path.join(app.config['PROFILE_FOLDER'], profile_name)
            profile.to_file(profile_path)

            return render_template('emr_profile.html', profile_url=url_for('static', filename=f'profile/{profile_name}'))
        else:
            return "❌ Invalid file. Please upload a CSV file."

    return render_template('emr_profile.html', profile_url=None)

@app.route('/emr-prediction', methods=['GET', 'POST'])
def emr_prediction():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename, {'csv'}):
            filename = str(uuid.uuid4()) + '.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            X = df.select_dtypes(include=['number']).fillna(0)

            predictions = model.predict(X)
            predicted_classes = (predictions > 0.5).astype(int)

            df['Prediction'] = predicted_classes
            table_html = df.to_html(classes='table table-bordered table-striped', index=False)

            return render_template('emr_prediction.html', table_html=table_html)
        else:
            return "❌ Invalid file. Please upload a CSV file."

    return render_template('emr_prediction.html', table_html=None)

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename, {'png', 'jpg', 'jpeg'}):
        return jsonify({"error": "❌ Vui lòng tải ảnh hợp lệ (.jpg, .png, .jpeg)"}), 400

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        label = "Nodule" if pred >= 0.5 else "Non-Nodule"
        confidence = float(pred if pred >= 0.5 else 1 - pred)

        return jsonify({"prediction": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
