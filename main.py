import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json  # <-- for class indices

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect the plant disease. The model will predict the disease and show the confidence level.")

# -------------------- Load Model --------------------
model_path = "plant_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file not found. Make sure plant_model.h5 is in the same folder as main.py.")

# -------------------- Load Class Indices --------------------
class_indices_path = "class_indices.json"
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
else:
    st.error("❌ Class indices file not found.")

# -------------------- File Uploader --------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file and os.path.exists(model_path) and os.path.exists(class_indices_path):
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]  # model output
    predicted_index = str(np.argmax(prediction))
    predicted_class = class_indices[predicted_index]  # map index to class name
    confidence = np.max(prediction) * 100

    # Display result
    st.markdown("### Prediction Result")
    if "healthy" in predicted_class.lower():
        st.success(f"**Predicted Disease:** {predicted_class}")
    else:
        st.error(f"**Predicted Disease:** {predicted_class}")

    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")
