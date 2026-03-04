import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Apple Disease Detector", layout="centered")
st.title("🍎 Apple Leaf Disease AI Detector")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "apple_disease_model.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

model = load_my_model()

# WAJIB sesuai urutan training dataset (abjad)
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

uploaded_file = st.file_uploader("📸 Upload gambar daun apel...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang di-upload", use_container_width=True)

    # ================= PREPROCESSING =================
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)

    # PENTING:
    # Kalau model lu SUDAH punya layers.Rescaling atau preprocess_input di dalamnya,
    # jangan tambah preprocessing lagi di sini.
    img_array = np.expand_dims(img_array, axis=0)

    # ================= PREDIKSI =================
    predictions = model.predict(img_array, verbose=0)[0]

    st.subheader("📊 Hasil Prediksi Model:")

    for i, class_name in enumerate(class_names):
        prob = float(predictions[i])
        st.write(f"**{class_name}** : {prob*100:.2f}%")
        st.progress(prob)

    max_idx = np.argmax(predictions)
    st.success(
        f"✅ Prediksi Terkuat: **{class_names[max_idx]}** "
        f"({predictions[max_idx]*100:.2f}%)"
    )