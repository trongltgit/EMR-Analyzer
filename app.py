import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from ydata_profiling import ProfileReport
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROFILE_FOLDER'] = 'static/profile'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'models/best_weights_model.keras'
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emr-profile', methods=['GET', 'POST'])
def emr_profile():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
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
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + '.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            # Preprocess: Drop non-numeric, fillna, convert to float
            X = df.select_dtypes(include=['number']).fillna(0)

            predictions = model.predict(X)
            predicted_classes = (predictions > 0.5).astype(int)

            df['Prediction'] = predicted_classes

            table_html = df.to_html(classes='table table-bordered table-striped', index=False)

            return render_template('emr_prediction.html', table_html=table_html)
        else:
            return "❌ Invalid file. Please upload a CSV file."

    return render_template('emr_prediction.html', table_html=None)

if __name__ == '__main__':
    app.run(debug=True)
