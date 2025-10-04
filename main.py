import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests  # 👈 Added this import

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the plant disease. The model will predict the disease and show the confidence level.")

# -------------------- Model Download & Load --------------------
model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"  # 👈 Replace YOUR_FILE_ID with your actual ID
model_path = "plant_model.h5"

# If model not found locally, download it
if not os.path.exists(model_path):
    st.info("📥 Downloading model from Google Drive...")
    with open(model_path, "wb") as f:
        f.write(requests.get(model_url).content)
    st.success("✅ Model downloaded from Google Drive!")

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# -------------------- File Uploader --------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_names = ["Apple___Black_rot", "Apple___Healthy", "Tomato___Late_blight"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show Results
    st.markdown("### Prediction Result")
    st.write(f"**Predicted Disease:** {predicted_class}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")
