import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import time
# Sử dụng log level của TensorFlow để tránh cảnh báo
import logging
tf.get_logger().setLevel(logging.ERROR)

# ========================= CONFIG UI ==========================
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered",
)

# Custom CSS for cleaner aesthetics, centering elements, and modern 'Tailwind-like' design
st.markdown("""
<style>
/* 1. Background and Typography */
.stApp {
    background-color: #f0fff0; /* Nền xanh lá cây nhạt */
    color: #1a1a1a;
    padding-top: 2rem;
}
h3, h2, h1 {
    color: #059669; /* Xanh lá cây chuyên nghiệp */
    font-weight: 700;
    text-align: center;
    white-space: nowrap; /* Đảm bảo tiêu đề không bị xuống dòng */
}
/* Login Card styling for visual separation */
.login-container {
    max-width: 500px;
    margin: 0 auto;
    padding: 30px;
    border-radius: 10px;
    background-color: #ffffff; /* Nền card trắng */
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1); /* Shadow hiện đại hơn */
}
/* Input fields style */
div.stTextInput>div>div>input {
    border-radius: 8px;
    border: 1px solid #d1d5db; /* Border nhạt hơn */
    padding: 12px;
}
/* Centering Logo/Images */
.stImage {
    text-align: center;
}
.stImage > img {
    display: inline-block;
    border-radius: 8px;
}
/* Button Styling (Modern Look) */
.stButton>button {
    background-color: #059669;
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    transition: background-color 0.3s, transform 0.1s;
    font-weight: 700;
    width: 100%;
    margin-top: 15px;
    border: none;
}
.stButton>button:hover {
    background-color: #047857;
    transform: translateY(-1px); /* Hiệu ứng nhấn nhẹ */
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

if not st.session_state.logged_in:
    # Start login-container block
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h2>🔒 Đăng nhập hệ thống</h2>", unsafe_allow_html=True)

    col_form, col_hint = st.columns([2, 1])

    with col_form:
        username_input = st.text_input("User ID", placeholder="Nhập ID", key="username_val")
        password_input = st.text_input("Password", type="password", placeholder="Nhập mật khẩu", key="password_val")
        login_btn = st.button("Đăng nhập")

    with col_hint:
        st.markdown("<h5 style='color: #059669;'>Gợi ý (Demo)</h5>", unsafe_allow_html=True)
        st.markdown(f"**ID:** `{USER}`")
        st.markdown(f"**Pass:** `{PASS}`")

    if login_btn:
        if username_input == USER and password_input == PASS:
            st.session_state.logged_in = True
            st.rerun() 
        else:
            st.error("Sai tài khoản hoặc mật khẩu!")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ========================= MAIN APP: HEADER & LOGO ======================

# Tiêu đề chính (đã fix lỗi xuống dòng)
st.header("🌿 Plant Disease Detection System")

# Centered Logo Display (FIXED: Thay 'auto' bằng số nguyên 1 để tránh lỗi TypeError)
logo_path = "assets/Logo_Marie_Curie.png" 
col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 2, 1])
with col_logo_2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=180)
    else:
        st.markdown("<div style='text-align: center; padding: 10px;'>*(Logo Placeholder)*</div>", unsafe_allow_html=True)
st.markdown("---")

# ========================= GOOGLE DRIVE MODEL HANDLING =================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1pLZYbUXHnoIEZEHrjg2Q-bj9Q47vOKh1"
MODEL_PATH = "plant_disease_Cnn.h5"

@st.cache_resource(show_spinner=False)
def load_model_from_drive():
    # Download the model if it doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Đang tải mô hình..."):
                time.sleep(1) 
                r = requests.get(MODEL_URL, stream=True)
                r.raise_for_status() 
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình: {e}")
            st.stop()
            
    # Load model (FIXED: Đã loại bỏ đoạn code gây lỗi AttributeError)
    with st.spinner("Đang load mô hình..."):
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model and cache it
model = load_model_from_drive()

# ========================= CLASS LABELS =====================
num_classes = model.output_shape[-1]

if num_classes == 2:
    classes = ["disease", "healthy"] 
else:
    classes = [f"class_{i}" for i in range(num_classes)]

# ========================= IMAGE UPLOAD & PREDICTION ======================
st.subheader("📸 Tải ảnh lá cây để nhận diện bệnh")

uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "jpeg", "png"])

def prepare(img):
    """Preprocesses the image: resize, normalize, and add batch dimension."""
    img = img.resize((224, 224)) # Model input size
    img = np.asarray(img) / 255.0 # Normalize
    if len(img.shape) == 2:  # Handle grayscale images
        img = np.stack((img,) * 3, axis=-1)
    return np.expand_dims(img, axis=0) # Add batch dimension

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Display image centered
    col_img_1, col_img_2, col_img_3 = st.columns([1, 2, 1])
    with col_img_2:
        st.image(img, caption="Ảnh đã tải lên", width=300)

    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang phân tích hình ảnh..."):
            if model is None:
                st.error("Mô hình chưa được tải thành công. Vui lòng thử lại.")
            else:
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
