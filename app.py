import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os
import gdown

# ──────────────────────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────────────────────

# Google Drive File ID (make sure sharing is set to "Anyone with the link")
file_id = "1ovY-0eB1u7D_XVlPqFH0QF13LdZgMuHY"
url = f"https://drive.google.com/uc?id={file_id}"
model_file = "Optimized_SVM_Pneumonia_Model.joblib"

# Download model if not already present
if not os.path.exists(model_file):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(url, model_file, quiet=False)

# Double-check file exists
if not os.path.exists(model_file):
    st.error("Model file not found. Please check your Google Drive link or file ID.")
    st.stop()

# Load PCA and SVM model
pca, model = joblib.load(model_file)

# ──────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────

st.title("🩺 Pneumonia Detection using SVM")

st.markdown("""
Upload a **chest X-ray** and this app will predict if the image indicates **Pneumonia** or is **Normal**.
""")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess for model
    image_array = np.array(image) / 255.0  # normalize
    image_flat = image_array.flatten().reshape(1, -1)
    image_pca = pca.transform(image_flat)

    # Predict
    prediction = model.predict(image_pca)
    label = "🦠 PNEUMONIA DETECTED" if prediction[0] == 1 else "✅ NORMAL"

    st.subheader("Prediction Result")
    st.success(label)
