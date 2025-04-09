import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load the model
model = joblib.load("model.joblib")

st.title("Pneumonia Detection using SVM")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image).flatten().reshape(1, -1)

    prediction = model.predict(image_np)
    result = "PNEUMONIA" if prediction[0] == 1 else "NORMAL"
    st.subheader(f"Prediction: {result}")
