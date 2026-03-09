import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setup Halaman
st.set_page_config(page_title="Apple Leaf Disease Detection", page_icon="🍎", layout="wide")

st.title("🍎 Apple Leaf Disease Detection")
st.markdown("Unggah gambar daun apel untuk mendeteksi penyakit secara otomatis menggunakan deep learning.")
st.divider()

# Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "apple_disease_robust.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

model = load_my_model()

# Urutan WAJIB sesuai training
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

# Database Penyakit
disease_info = {
    "Apple___Apple_scab": {
        "nama_keren": "Apple Scab",
        "status": "Terdeteksi Penyakit",
        "deskripsi": "Infeksi jamur Venturia inaequalis yang ditandai dengan bercak gelap bertekstur kasar pada permukaan daun.",
        "solusi": "Aplikasikan fungisida berbahan aktif Captan atau Mancozeb secara rutin. Bersihkan daun yang gugur untuk mencegah penyebaran spora."
    },
    "Apple___Black_rot": {
        "nama_keren": "Black Rot",
        "status": "Terdeteksi Penyakit",
        "deskripsi": "Penyakit jamur yang ditandai dengan bercak coklat melingkar konsentris, dapat menyerang daun hingga buah.",
        "solusi": "Pangkas dan musnahkan bagian tanaman yang terinfeksi. Gunakan fungisida berbahan aktif Myclobutanil sebagai tindakan preventif."
    },
    "Apple___Cedar_apple_rust": {
        "nama_keren": "Cedar Apple Rust",
        "status": "Terdeteksi Penyakit",
        "deskripsi": "Infeksi jamur Gymnosporangium juniperi-virginianae yang menghasilkan bercak kuning hingga oranye cerah pada permukaan daun.",
        "solusi": "Pangkas ranting yang terinfeksi dan hindari penanaman pohon apel berdekatan dengan pohon Cedar sebagai inang perantara."
    },
    "Apple___healthy": {
        "nama_keren": "Healthy",
        "status": "Daun Sehat",
        "deskripsi": "Tidak ditemukan indikasi penyakit pada daun. Kondisi tanaman dalam keadaan baik.",
        "solusi": "Pertahankan jadwal penyiraman, pemupukan, dan pemangkasan rutin untuk menjaga kesehatan tanaman."
    }
}

# Confidence Threshold
CONFIDENCE_THRESHOLD = 0.70

uploaded_file = st.file_uploader("📸 Unggah gambar daun apel (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📸 Input Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 Analysis Result")

        # Preprocessing
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediksi
        predictions = model.predict(img_array, verbose=0)[0]
        max_idx = np.argmax(predictions)
        akurasi = predictions[max_idx] * 100

        # Confidence Threshold Check
        if akurasi < CONFIDENCE_THRESHOLD * 100:
            st.warning("⚠️ Model tidak cukup yakin dengan gambar ini. Pastikan gambar yang diunggah adalah daun apel yang jelas.")
        else:
            prediksi_asli = class_names[max_idx]
            info = disease_info[prediksi_asli]

            if info["status"] == "Daun Sehat":
                st.success(f"✅ **{info['nama_keren']}** — No Disease Detected ({akurasi:.2f}%)")
            else:
                st.error(f"⚠️ **{info['nama_keren']}** — Disease Detected ({akurasi:.2f}%)")

            st.info(f"**Diagnosis:** {info['deskripsi']} \n\n**Solusi:** {info['solusi']}")

        st.divider()

        # Detail Probabilitas
        st.subheader("🔍 Confidence Score")
        for i, class_name in enumerate(class_names):
            prob = float(predictions[i])
            nama_rapi = disease_info[class_name]["nama_keren"]
            st.write(f"{nama_rapi}: {prob*100:.2f}%")
            st.progress(prob)