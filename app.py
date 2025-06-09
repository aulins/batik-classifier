import os
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Inisialisasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan label_map
model = load_model('model/model_ds82_normalized_augmentedClean_greyscale_resized_lr00005.h5')
with open('model/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocessing
            img = load_img(filepath, target_size=(224, 224))  # Jangan grayscale
            img = img_to_array(img)                           # (224, 224, 3)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)                 # (1, 224, 224, 3)

            # Prediksi
            pred_probs = model.predict(img)
            pred_idx = np.argmax(pred_probs)
            pred_label = inv_label_map[pred_idx]
            confidence = float(pred_probs[0][pred_idx])

            return render_template('index.html', filename=file.filename, label=pred_label, confidence=confidence)

    return render_template('index.html')

# Route menampilkan gambar upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
