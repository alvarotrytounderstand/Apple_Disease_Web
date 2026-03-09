# 🍎 Apple Leaf Disease Detection

Aplikasi berbasis web untuk mendeteksi penyakit pada daun apel secara *real-time* menggunakan model Deep Learning (MobileNetV2). Proyek ini dibangun menggunakan **TensorFlow/Keras** dan di-*deploy* sebagai aplikasi interaktif menggunakan **Streamlit**.

🔗 **[Coba Aplikasinya di Sini](https://applediseaseweb.streamlit.app/)**

---

## 🌟 Fitur Utama

- **Deteksi 4 Kondisi Daun** — Mengenali daun Healthy, Apple Scab, Black Rot, dan Cedar Apple Rust.
- **Advanced Data Augmentation** — Model dilatih dengan teknik augmentasi tingkat lanjut (Flip, Rotation, Zoom, Translation, Brightness, & Contrast) untuk meningkatkan generalisasi pada kondisi nyata.
- **Actionable Insights** — Setiap hasil prediksi disertai deskripsi penyakit dan rekomendasi penanganan.
- **Confidence Score** — Menampilkan probabilitas prediksi untuk setiap kelas secara transparan.
- **Smart Warning System** — Indikator visual dinamis: merah untuk penyakit terdeteksi, hijau untuk kondisi sehat.
- **Confidence Threshold** — Model akan memberikan peringatan jika tingkat keyakinan prediksi di bawah 70%.

---

## 🚀 Performa Model

Model menggunakan arsitektur **MobileNetV2** dengan pendekatan *Transfer Learning* dan *Fine-Tuning*:

| Metrik | Nilai |
|---|---|
| Training Accuracy | 98.09% |
| Validation Accuracy | 99.81% |
| Validation Loss | 0.0057 |
| Precision / Recall / F1 | 1.00 (per semua kelas) |

---

## 🛠️ Tech Stack

- **Machine Learning:** TensorFlow, Keras
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pillow (PIL)
- **Language:** Python 3.x

---

## 💻 Instalasi Lokal

1. **Clone repository:**
   ```bash
   git clone https://github.com/alvarotrytounderstand/Apple_Disease_Web.git
   cd Apple_Disease_Web
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

4. Buka browser dan akses `http://localhost:8501`.

---

## 📂 Struktur Direktori

```
Apple_Disease_Web/
├── app.py                        # Skrip utama Streamlit
├── apple_disease_robust.keras    # Model MobileNetV2 hasil training
├── requirements.txt              # Daftar dependensi
└── notebook/                     # Jupyter Notebook (eksplorasi, training, evaluasi)
```