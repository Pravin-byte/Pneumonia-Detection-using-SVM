from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)

# Load PCA and model
pca, model = joblib.load("Optimized_SVM_Pneumonia_Model.joblib")

@app.route("/")
def home():
    return "🩺 Pneumonia Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).resize((128, 128)).convert('RGB')
    image = np.array(image) / 255.0
    image = image.flatten().reshape(1, -1)

    image_pca = pca.transform(image)
    prediction = model.predict(image_pca)
    label = "Pneumonia" if prediction[0] == 1 else "Normal"
    
    return jsonify({"prediction": label})
