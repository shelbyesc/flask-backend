from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model at startup
model = load_model("models/xray_model.h5")

@app.route('/')
def home():
    return jsonify(message="Flask backend is running. Model loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify(error="Missing image in request"), 400

        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  # Adjust this size if your model expects something else

        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize if needed
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        return jsonify(prediction=prediction.tolist())
    except Exception as e:
        return jsonify(error=str(e)), 500

# Optional: only needed for local testing
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
