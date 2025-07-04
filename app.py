import os
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Path to save the model locally
MODEL_PATH = "models/xray_model.h5"  # Local path for storing the model file

# Google Drive file ID (replace with your actual file ID)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1sth_LGZYfe7Gpc5TYmKP0nsWoZ27PHo3"

# Check if the model exists locally, if not, download it from Google Drive
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)  # Create models folder if it doesn't exist
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as model_file:
        model_file.write(response.content)
    print("Model downloaded successfully.")

# Load the model (this is done once at app startup)
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return jsonify(message="Flask backend is running. Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify(error="Missing image in request"), 400

        # Get the uploaded image file
        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))  # Resize image to match model input size (adjust this if necessary)

        # Convert the image to an array and normalize it
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(img_array)

        # Return the prediction as a JSON response
        return jsonify(prediction=prediction.tolist())
    
    except Exception as e:
        return jsonify(error=str(e)), 500

# Optional: only needed for local testing
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the port provided by Render or default to 5000
    app.run(host='0.0.0.0', port=port)
