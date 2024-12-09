import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Parameter aplikasi
IMAGE_SIZE = 224
CLASSES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("saved_model_potato_CNN.keras")
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

model = load_model()

# Fungsi prediksi
def predict(image):
    try:
        image_array = img_to_array(image)
        image_array = tf.image.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
        prediction = model.predict(image_array)
        predicted_class = CLASSES[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None

# Fungsi untuk menampilkan EDA
def eda(directory):
    try:
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Informasi Dataset:")

        # Hitung jumlah gambar per kelas
        data_info = {}
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                data_info[folder] = len(os.listdir(folder_path))

        if data_info:
            # Tampilkan tabel
            st.table({"Kelas": list(data_info.keys()), "Jumlah Gambar": list(data_info.values())})

            # Visualisasi jumlah gambar per kelas
            fig, ax = plt.subplots()
            ax.bar(data_info.keys(), data_info.values(), color=["blue", "green", "orange", "red"])
            ax.set_title("Distribusi Gambar per Kelas")
            ax.set_xlabel("Kelas")
            ax.set_ylabel("Jumlah Gambar")
            st.pyplot(fig)
        else:
            st.error("Folder dataset kosong atau tidak valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan EDA: {e}")

# Streamlit UI
def main():
    st.title("Prediksi Penyakit Daun Anggur")
    st.write("Unggah gambar daun anggur untuk memprediksi jenis penyakitnya.")

    menu = ["Prediksi", "EDA"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediksi":
        st.subheader("Prediksi Penyakit Daun")
        uploaded_file = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang diunggah", use_column_width=True)

                if st.button("Prediksi"):
                    predicted_class, confidence = predict(image)
                    if predicted_class:
                        st.success(f"Kelas Prediksi: {predicted_class}")
                        st.info(f"Tingkat Kepercayaan: {confidence}%")
                    else:
                        st.error("Gagal melakukan prediksi.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

    elif choice == "EDA":
        st.subheader("Exploratory Data Analysis (EDA)")
        dataset_path = st.text_input("Masukkan path dataset", "/content/primer+sekunder/primer+sekunder/Train")

        if dataset_path:
            if os.path.exists(dataset_path):
                eda(dataset_path)
            else:
                st.error("Path dataset tidak valid atau tidak ditemukan.")
        else:
            st.info("Masukkan path dataset untuk melakukan analisis.")

if __name__ == "__main__":
    main()
