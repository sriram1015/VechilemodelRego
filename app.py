from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('ivamicro.h5', compile=False)

# Ensure the 'static/uploads' directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Adjust based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Mapping class indices to class names (update this according to your model)
class_names = {0: 'Audi', 1: 'Hyundai Creta', 2: 'Mahindra Scorpio',3:"Rolls Royce",4:"Swift",5:"Tata Safari",6:"Toyota Innova"}

@app.route('/')
def index():
    return render_template('index.html', prediction=None, img_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file to the static/uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        processed_image = load_and_preprocess_image(file_path)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names.get(predicted_class, "Unknown")

        # Return the image path relative to the static folder
        return render_template('index.html', prediction=predicted_class_name, img_path=f'uploads/{filename}')

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
