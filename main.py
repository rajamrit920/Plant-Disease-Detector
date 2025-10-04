import streamlit as st
import gdown
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the plant disease. The model will predict the disease and show the confidence level.")

# -------------------- Model Setup --------------------
model_path = "plant_model.h5"
file_id = "19qE1l-lkAAbyW5dfbnOOjKzQDFFg7HUg"  # Replace with your Google Drive file ID
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if it does not exist
if not os.path.exists(model_path):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

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
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    # Predict
    class_names = ["Apple___Black_rot", "Apple___Healthy", "Tomato___Late_blight"]
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.markdown("### Prediction Result")
    st.write(f"**Predicted Disease:** {predicted_class}")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")
