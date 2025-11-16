import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# ========================= CONFIG UI ==========================
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered",
)

# Background CSS
page_bg = """
<style>
body {
    background-color: #E2F6E9;
}
.block-container {
    padding-top: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ========================= LOGIN ==============================
st.markdown("## 🔒 Đăng nhập hệ thống")

USER = "user_demo"
PASS = "Test@123456"

username = st.text_input("User ID")
password = st.text_input("Password", type="password")
login_btn = st.button("Đăng nhập")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if login_btn:
    if username == USER and password == PASS:
        st.session_state.logged_in = True
        st.success("Đăng nhập thành công!")
    else:
        st.error("Sai tài khoản hoặc mật khẩu!")

if not st.session_state.logged_in:
    st.stop()

# ========================= LOGO ===============================
st.markdown("### 🌿 Plant Disease Detection System")
logo_path = "assets/Logo_Marie_Curie.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=180)

st.markdown("---")

# ========================= GOOGLE DRIVE MODEL DOWNLOAD =========================

MODEL_URL = "https://drive.google.com/uc?export=download&id=1pLZYbUXHnoIEZEHrjg2Q-bj9Q47vOKh1"
MODEL_PATH = "plant_disease_Cnn.h5"

@st.cache_resource
def load_model_from_drive():
    # Nếu file chưa tồn tại → tải từ Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔽 Đang tải mô hình từ Google Drive..."):
            r = requests.get(MODEL_URL)
            open(MODEL_PATH, "wb").write(r.content)
            st.success("✔ Tải mô hình thành công!")

    # Load model
    with st.spinner("🔧 Đang load mô hình AI..."):
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✔ Mô hình đã sẵn sàng!")
    return model

model = load_model_from_drive()

# ========================= AUTO CLASS LOADING ======================

num_classes = model.output_shape[-1]

if num_classes == 2:
    classes = ["healthy", "disease"]
else:
    classes = [f"class_{i}" for i in range(num_classes)]

st.info(f"Classes loaded: {classes}")

# ========================= IMAGE UPLOAD ============================
st.subheader("📸 Tải ảnh lá cây để nhận diện bệnh")

uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "jpeg", "png"])

def prepare(img):
    img = img.resize((224, 224))
    img = np.asarray(img) / 255.0
    return np.expand_dims(img, axis=0)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", width=300)

    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang phân tích hình ảnh..."):
            x = prepare(img)
            pred = model.predict(x)
            class_id = np.argmax(pred)
            confidence = np.max(pred)

        st.success(f"🌿 **Kết quả:** {classes[class_id]}")
        st.info(f"Độ tin cậy: {confidence * 100:.2f}%")
