import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setup Halaman
st.set_page_config(page_title="Apple Disease AI", page_icon="🍎", layout="wide")

st.title("🍎 Apple Leaf Disease AI Sultan")
st.markdown("Deteksi penyakit pada daun apel dalam hitungan detik beserta solusinya!")
st.divider()

# Load Model
@st.cache_resource
def load_my_model():
    # Pastikan nama file ini sesuai sama yang lu save terakhir di Jupyter Notebook!
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

# Database Penyakit & Resep Obat (ACTIONABLE INSIGHTS) 💊
disease_info = {
    "Apple___Apple_scab": {
        "nama_keren": "Kudis Apel (Apple Scab)",
        "status": "Bahaya",
        "deskripsi": "Infeksi jamur yang bikin bercak hitam/zaitun kasar di permukaan daun.",
        "solusi": "Semprotkan fungisida (seperti Captan atau Mancozeb) dan segera bersihkan daun yang gugur agar jamur tidak menyebar."
    },
    "Apple___Black_rot": {
        "nama_keren": "Busuk Hitam (Black Rot)",
        "status": "Bahaya",
        "deskripsi": "Penyakit jamur mematikan yang bikin bercak coklat melingkar seperti mata katak.",
        "solusi": "Potong dan bakar bagian yang terinfeksi. Gunakan fungisida berbahan aktif Myclobutanil."
    },
    "Apple___Cedar_apple_rust": {
        "nama_keren": "Karat Cedar (Cedar Apple Rust)",
        "status": "Bahaya",
        "deskripsi": "Jamur karat yang bikin bercak kuning/oranye cerah mencolok di atas daun.",
        "solusi": "Pangkas ranting yang terinfeksi dan hindari menanam pohon apel di dekat pohon Cedar."
    },
    "Apple___healthy": {
        "nama_keren": "Daun Sehat (Healthy)",
        "status": "Aman",
        "deskripsi": "Mantap! Daun apel dalam kondisi prima dan bebas dari penyakit.",
        "solusi": "Pertahankan rutinitas penyiraman dan pemupukan yang baik."
    }
}

uploaded_file = st.file_uploader("📸 Upload gambar daun apel...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # BIKIN LAYOUT SPLIT KIRI-KANAN Biar Elegan
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📸 Gambar Daun")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, border=True)

    with col2:
        st.subheader("📊 Hasil Analisis AI")
        
        # Preprocessing
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array, verbose=0)[0]
        max_idx = np.argmax(predictions)
        
        # Ambil data penyakit dari dictionary
        prediksi_asli = class_names[max_idx]
        info = disease_info[prediksi_asli]
        akurasi = predictions[max_idx] * 100

        # WARNING SYSTEM 🚦
        if info["status"] == "Aman":
            st.success(f"✅ **{info['nama_keren']}** ({akurasi:.2f}%)")
        else:
            st.error(f"⚠️ **{info['nama_keren']}** ({akurasi:.2f}%)")
            
        # Tampilkan Resep Obat
        st.info(f"**Diagnosis:** {info['deskripsi']} \n\n**Solusi:** {info['solusi']}")

        st.divider()
        
        # Detail Probabilitas (Biar kelihat pro)
        st.markdown("**🔍 Detail Probabilitas:**")
        for i, class_name in enumerate(class_names):
            prob = float(predictions[i])
            nama_rapi = disease_info[class_name]["nama_keren"]
            st.write(f"{nama_rapi}: {prob*100:.2f}%")
            st.progress(prob)