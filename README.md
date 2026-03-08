# 🍎 Apple Leaf Disease AI Detector

Aplikasi berbasis web (*Dashboard*) untuk mendeteksi penyakit pada daun apel secara *real-time* menggunakan model Deep Learning (MobileNetV2). Proyek ini dibangun menggunakan **TensorFlow/Keras** dan di-*deploy* sebagai aplikasi interaktif menggunakan **Streamlit**.

🔗 **[Coba Aplikasinya Langsung di Sini]** *https://applediseaseweb.streamlit.app/*

## 🌟 Fitur Utama (Features)
- **Deteksi 4 Kondisi Daun:** Mengenali daun Sehat (*Healthy*), Kudis Apel (*Apple Scab*), Busuk Hitam (*Black Rot*), dan Karat Cedar (*Cedar Apple Rust*).
- **Robust AI (Tahan Banting):** Model dilatih menggunakan teknik *Data Augmentation* tingkat lanjut (Flip, Rotation, Zoom, Translation, Brightness, & Contrast) sehingga kebal terhadap variasi pencahayaan dan sudut foto di dunia nyata.
- **Actionable Insights:** Tidak hanya memprediksi, aplikasi juga memberikan deskripsi penyakit dan **resep/solusi penanganan** yang harus dilakukan oleh petani atau pengguna.
- **Smart Warning System:** Indikator warna dinamis (Merah untuk bahaya/sakit, Hijau untuk aman/sehat).
- **Live Camera / File Upload:** Mendukung input gambar melalui unggahan file maupun jepretan kamera langsung (*Responsive* di HP).

## 🚀 Performa Model (Model Performance)
Model ini menggunakan arsitektur **MobileNetV2** (Transfer Learning) dengan hasil evaluasi yang sangat memuaskan dan **bebas dari overfitting**:
- **Training Accuracy:** `98.09%`
- **Validation Accuracy:** `99.81%`
- **Validation Loss:** `0.0057`
- **Precision, Recall, F1-Score:** `1.00` (Sempurna di semua metrik pengujian).

## 🛠️ Teknologi yang Digunakan (Tech Stack)
- **Machine Learning:** TensorFlow, Keras (Format `.keras` modern)
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pillow (PIL)
- **Bahasa:** Python 3.x

## 💻 Cara Menjalankan di Komputer Sendiri (Local Installation)

Jika Anda ingin mencoba menjalankan kode ini di laptop/komputer Anda, ikuti langkah-langkah berikut:

1. **Clone Repository ini:**
   git clone https://github.com/alvarotrytounderstand/Apple_Disease_Web.git
   cd Apple_Disease_Web


2. **Install Dependensi:**
    Pastikan Anda sudah menginstal Python. Lalu jalankan perintah ini di terminal:
    pip install tensorflow streamlit numpy pillow

3. **Jalankan Streamlit App:**
    streamlit run app.py

4. Buka browser dan akses http://localhost:8501.

## 📂 Struktur Direktori
- `app.py` : Skrip utama untuk menjalankan dashboard Streamlit.
- `apple_disease_robust.keras` : Model Deep Learning (MobileNetV2) yang sudah dilatih dan siap pakai (Inference).
- `notebook/` : (Opsional) Folder berisi file Jupyter Notebook (.ipynb) tempat proses eksplorasi data, training, dan evaluasi model dilakukan.