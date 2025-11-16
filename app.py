import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import time
import logging
import base64
from io import BytesIO

tf.get_logger().setLevel(logging.ERROR)

# ========================= CONFIG ==========================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# ========================= CSS ==========================
st.markdown("""
<style>
    :root {
        --primary: #2e7d32;
        --light-bg: #f8fff9;
        --success: #d4edda;
        --warning: #fff3cd;
    }
    .stApp {
        background-color: var(--light-bg);
        font-family: 'Segoe UI', sans-serif;
    }
    .header-container {
        background: white;
        padding: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-container img {
        height: 60px;
        width: auto;
        object-fit: contain;
    }
    h1, h2, h3 {
        color: var(--primary) !important;
        text-align: center;
        font-weight: 600;
    }
    .main-card, .login-container {
        max-width: 600px;
        margin: 0 auto 2rem;
        padding: 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .stTextInput > div > div > input,
    .stFileUploader > div > div {
        background: white !important;
        color: #1a1a1a !important;
        border: 1.5px solid #ccc !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    .stButton > button {
        background: var(--primary) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        width: 100%;
        border: none !important;
        margin-top: 1rem;
    }
    .uploaded_img img {
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-radius: 12px;
    }
    .stSuccess {
        background: var(--success);
        border-left: 5px solid #28a745;
        padding: 12px;
        border-radius: 8px;
        color: #155724;
    }
    .stWarning {
        background: var(--warning);
        border-left: 5px solid #ffc107;
        padding: 12px;
        border-radius: 8px;
        color: #856404;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
        background: white;
    }
    .footer a { color: var(--primary); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ========================= LOGIN ==========================
USER, PASS = "user_demo", "Test@123456"

def render_header():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    logo_path = "assets/Logo_Marie_Curie.png"
    if os.path.exists(logo_path):
        try:
            img = Image.open(logo_path).resize((180, 60), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_str}" style="height:60px;">', unsafe_allow_html=True)
        except:
            st.markdown("**Logo Marie Curie**", unsafe_allow_html=True)
    else:
        st.markdown("**Logo Marie Curie**", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="footer">
        <p><strong>Liên hệ:</strong> Công ty TNHH MTV Minh Trí và những người bạn Marie Curie<br>
        159 Nam Kỳ Khởi Nghĩa, Phường Xuân Hòa, Tp. Hồ Chí Minh<br>
        Lại Nguyễn Minh Trí - <a href="mailto:laingminhtri@gmail.com">laingminhtri@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_header()
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h1>Hệ thống Phát hiện Bệnh Cây bằng AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;'>Ứng dụng nhận diện bệnh trên lá cây</p>", unsafe_allow_html=True)

    with st.form("login"):
        st.text_input("ID người dùng", placeholder="user_demo", key="user", label_visibility="collapsed")
        st.text_input("Mật khẩu", type="password", placeholder="Test@123456", key="pwd", label_visibility="collapsed")
        login = st.form_submit_button("Đăng nhập")

    if login:
        if st.session_state.user == USER and st.session_state.pwd == PASS:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Sai tài khoản hoặc mật khẩu!")

    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()
    st.stop()

# ========================= LOAD MODEL from GitHub ==========================

MODEL_PATH = "plant_disease_mobilenet.h5"

# 👉👉 Hãy thay URL này bằng link RAW GitHub thật của bạn
MODEL_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/model/plant_disease_mobilenet.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Đang tải mô hình từ GitHub..."):
            r = requests.get(MODEL_URL)

            if r.status_code != 200:
                st.error("❌ Không thể tải mô hình từ GitHub. Kiểm tra lại MODEL_URL.")
                st.stop()

            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    with st.spinner("Đang load mô hình..."):
        return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ========================= CLASS LABELS ==========================
classes = ["BỆNH", "KHỎE MẠNH"]

# ========================= PREPROCESS ==========================
def prepare(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = x[:, :, ::-1]  # RGB → BGR
    x /= 255.0
    return np.expand_dims(x, axis=0)

# ========================= MAIN UI ==========================
render_header()
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>Phát hiện Bệnh Cây</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Tải lên ảnh lá cây để nhận diện bệnh</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<div class='uploaded_img'>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Dự đoán", use_container_width=True):
        with st.spinner("Đang phân tích..."):
            x = prepare(img)
            pred = model.predict(x)[0]
            conf = float(np.max(pred))
            label = classes[np.argmax(pred)]

        if conf > 0.7:
            st.balloons()
            st.markdown(f'<div class="stSuccess"><strong>Kết quả: {label}</strong></div>', unsafe_allow_html=True)
            st.metric("Độ tin cậy", f"{conf*100:.1f}%")
        else:
            st.markdown(
                f'<div class="stWarning">Không rõ ràng: {label} ({conf*100:.1f}%). Hãy chụp lại rõ hơn.</div>',
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)
render_footer()
