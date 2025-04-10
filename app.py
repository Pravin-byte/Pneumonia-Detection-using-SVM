import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os
import gdown

# Model filename
model_file = "Optimized_SVM_Pneumonia_Model.joblib"

# Download from Google Drive if not present
file_id = "1ovY-0eB1u7D_XVlPqFH0QF13LdZgMuHY"  # <-- Replace with your Google Drive file ID
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_file):
    st.info("Downloading model from Google Drive...")
    gdown.download(url, model_file, quiet=False)

# Load PCA and SVM model
pca, model = joblib.load(model_file)

st.title("🩺 Pneumonia Detection using SVM")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    image_np = np.array(image) / 255.0
    image_np = image_np.flatten().reshape(1, -1)

    # PCA transform and predict
    image_pca = pca.transform(image_np)
    prediction = model.predict(image_pca)

    # Show result
    result = "🦠 PNEUMONIA DETECTED" if prediction[0] == 1 else "✅ NORMAL"
    st.subheader(f"Prediction: {result}")
