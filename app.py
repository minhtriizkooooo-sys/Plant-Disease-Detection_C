import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import time

# ========================= CONFIG UI ==========================
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered",
)

# Custom CSS for aesthetics, centering logo, and hiding Streamlit style elements
st.markdown("""
<style>
/* 1. Background and Typography */
.stApp {
    background-color: #E2F6E9; /* Light Mint Green */
    color: #1a1a1a;
    padding-top: 2rem;
}
h3, h2 {
    color: #047857; /* Dark Green */
    font-weight: 700;
    text-align: center; /* Centering titles */
}
/* 2. Centering Logo/Images */
.stImage {
    text-align: center;
}
.stImage > img {
    display: inline-block;
}
/* 3. Button Styling */
.stButton>button {
    background-color: #059669; /* Medium green */
    color: white;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    transition: background-color 0.3s;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #047857; /* Darker green on hover */
}
</style>
""", unsafe_allow_html=True)

# Define login credentials
USER = "user_demo"
PASS = "Test@123456"

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ========================= LOGIN PAGE =============================

# Only show login inputs if not logged in
if not st.session_state.logged_in:
    st.markdown("## 🔒 Đăng nhập hệ thống", unsafe_allow_html=True)

    # Hiển thị username và password mặc định dưới dạng placeholder
    username_input = st.text_input("User ID", value=USER, key="username_val")
    password_input = st.text_input("Password", type="password", value=PASS, key="password_val")
    login_btn = st.button("Đăng nhập")

    if login_btn:
        if username_input == USER and password_input == PASS:
            st.session_state.logged_in = True
            st.success("Đăng nhập thành công!")
            # Sau khi đăng nhập thành công, xóa nội dung hiển thị trong input 
            st.session_state.username_val = ""
            st.session_state.password_val = ""
            st.rerun() # Rerun để chuyển sang trang chính
        else:
            st.error("Sai tài khoản hoặc mật khẩu!")
    st.stop() # Dừng tại đây nếu chưa đăng nhập

# Nếu đã đăng nhập, code chạy tiếp từ đây

# ========================= MAIN APP: LOGO & HEADER ======================

# Centered Logo Display
col1, col2, col3 = st.columns([1, 2, 1])
logo_path = "assets/Logo_Marie_Curie.png"
with col2:
    st.markdown("### 🌿 Plant Disease Detection System", unsafe_allow_html=True)
    if os.path.exists(logo_path):
        st.image(logo_path, width=180)
    else:
        st.markdown("*(Logo Placeholder)*")
st.markdown("---")

# ========================= GOOGLE DRIVE MODEL =================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1pLZYbUXHnoIEZEHrjg2Q-bj9Q47vOKh1"
MODEL_PATH = "plant_disease_Cnn.h5"

@st.cache_resource(show_spinner=False)
def load_model_from_drive():
    # Tải model nếu chưa có
    if not os.path.exists(MODEL_PATH):
        try:
            # Ẩn thông báo success/loading để giao diện sạch hơn
            with st.spinner("Đang tải mô hình..."):
                r = requests.get(MODEL_URL, stream=True)
                r.raise_for_status() 
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {e}")
            st.stop()
            
    # Load model
    with st.spinner("Đang load mô hình..."):
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model_from_drive()

# ========================= CLASS LOADING (FIXED) =====================
num_classes = model.output_shape[-1]

if num_classes == 2:
    # ĐÃ SỬA: Thứ tự nhãn là 'disease' (Index 0) và 'healthy' (Index 1) 
    # để khắc phục lỗi đảo ngược kết quả.
    classes = ["disease", "healthy"] 
else:
    classes = [f"class_{i}" for i in range(num_classes)]

# ========================= IMAGE UPLOAD & PREDICTION ======================
st.subheader("📸 Tải ảnh lá cây để nhận diện bệnh")

uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "jpeg", "png"])

def prepare(img):
    img = img.resize((224, 224))
    img = np.asarray(img) / 255.0
    if len(img.shape) == 2:  # nếu ảnh grayscale, convert thành 3 channels
        img = np.stack((img,) * 3, axis=-1)
    return np.expand_dims(img, axis=0)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", width=300)

    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang phân tích hình ảnh..."):
            x = prepare(img)
            pred = model.predict(x)
            
            # Extract results
            class_id = int(np.argmax(pred))
            confidence = float(np.max(pred))
            
        # Display results
        if confidence * 100 > 70:
            st.balloons()
            st.success(f"🌿 **Kết quả:** {classes[class_id].upper()}")
            st.metric(label="Độ tin cậy", value=f"{confidence * 100:.2f}%")
        else:
            st.warning(f"Kết quả không rõ ràng. Kết quả tốt nhất: {classes[class_id]} với độ tin cậy {confidence * 100:.2f}%.")
