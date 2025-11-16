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

# ========================= CONFIG UI ==========================
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered",
)

# ========================= CUSTOM CSS & STYLES ==========================
# Áp dụng mã màu yêu cầu: --primary-green: #2e7d32 và --light-green-bg: #e8f5e9
st.markdown("""
<style>
/* Định nghĩa biến màu dựa trên yêu cầu của bạn */
:root {
    --primary-green: #2e7d32; /* Xanh lá đậm */
    --light-green-bg: #e8f5e9; /* Nền xanh lá nhạt, đẹp hơn */
    --accent-green: #4CAF50; /* Xanh lá cây nhấn */
    --dark-green-hover: #1b5e20; /* Màu khi hover nút */
}

/* 1. Global Background */
.stApp {
    background-color: var(--light-green-bg); 
    font-family: 'Inter', sans-serif;
    color: #1a1a1a;
    padding-top: 0;
}

/* 2. Header Style: White background, shadow */
.header-container {
    background-color: white;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
    padding: 1rem 0;
    margin-bottom: 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
}
.header-container img {
    max-height: 80px; 
    width: auto;
    object-fit: contain;
}

/* 3. Typography */
h3, h2, h1 {
    color: var(--primary-green);
    font-weight: 700;
    text-align: center;
    white-space: nowrap; 
}
h2 { font-size: 1.5rem; }
h1 { font-size: 2rem; }

/* 4. Main Card/Login Card styling */
.main-card, .login-container {
    max-width: 600px;
    margin: 0 auto 3rem auto;
    padding: 30px;
    border-radius: 12px;
    background-color: #ffffff; 
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); 
}

/* 5. Input fields style */
div.stTextInput>div>div>input, div.stFileUploader > label + div {
    border-radius: 8px;
    border: 1px solid #d1d5db; 
    padding: 10px;
    transition: all 0.15s;
}
div.stTextInput>div>div>input:focus, div.stFileUploader > label + div:focus {
    border-color: var(--primary-green);
    box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.5);
}

/* 6. Button Styling (Primary Green) */
.stButton>button {
    background-color: var(--primary-green);
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    transition: background-color 0.3s, transform 0.1s;
    font-weight: 600;
    width: 100%;
    margin-top: 15px;
    border: none;
}
.stButton>button:hover {
    background-color: var(--dark-green-hover); 
    transform: translateY(-1px); 
}

/* 7. Result Display Styling */
.stSuccess { background-color: #e6ffe6; border-left: 5px solid var(--accent-green); padding: 10px; border-radius: 5px; }
.stWarning { background-color: #fffbe6; border-left: 5px solid #ffc107; padding: 10px; border-radius: 5px; }
.stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-top: 15px; background-color: #f9f9f9; }

/* 8. Footer Style */
.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid #e0e0e0;
    color: #757575;
    font-size: 0.8rem;
    background-color: white;
}
.footer a {
    color: var(--primary-green);
}
</style>
""", unsafe_allow_html=True)

# Define login credentials
USER = "user_demo"
PASS = "Test@123456"

# ========================= HEADER COMPONENT ==========================

def render_header():
    """Hiển thị Header và Logo, sử dụng Base64 để nhúng ảnh (không dùng placeholder)."""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    logo_path = "assets/Logo_Marie_Curie.png"
    
    logo_html = ""

    if os.path.exists(logo_path):
        try:
            img = Image.open(logo_path)
            # Convert image to Base64
            buffered = BytesIO()
            img.save(buffered, format="PNG") 
            img_str = base64.b64encode(buffered.getvalue()).decode()
            logo_src = f"data:image/png;base64,{img_str}"
            logo_alt = "Logo Marie Curie"
            
            logo_html = f"""
                <div style='text-align: center;'>
                    <img src='{logo_src}' 
                         alt='{logo_alt}' 
                         class='max-h-20 w-auto object-contain'>
                </div>
            """
        except Exception as e:
            # Nếu xảy ra lỗi khi tải file, in lỗi ra console và hiển thị chữ thay thế
            logging.error(f"Lỗi khi load logo từ path {logo_path}: {e}")
            logo_html = f"""
                <div style='text-align: center; color: var(--primary-green); font-size: 1.5rem; font-weight: bold;'>
                    Logo Marie Curie (Tải ảnh lỗi)
                </div>
            """
    else:
        # Nếu không tìm thấy file, hiển thị chữ thay thế
        logo_html = f"""
            <div style='text-align: center; color: var(--primary-green); font-size: 1.5rem; font-weight: bold;'>
                Logo Marie Curie (File không tồn tại)
            </div>
        """
            
    with col_c:
        st.markdown(logo_html, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= FOOTER COMPONENT ==========================

def render_footer():
    """Hiển thị thông tin liên hệ ở cuối trang."""
    footer_html = """
    <div class="footer">
        <p><strong>Liên hệ:</strong> Công ty TNHH MTV Minh Trí và những người bạn Marie Curie<br>
        159 Nam Kỳ Khởi Nghĩa, Phường Xuân Hòa, Tp. Hồ Chí Minh<br>
        Lại Nguyễn Minh Trí - <a href="mailto:laingminhtri@gmail.com" style="color: var(--primary-green);">laingminhtri@gmail.com</a></p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ========================= LOGIN PAGE =============================

if not st.session_state.logged_in:
    render_header()
    
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <h1 class="text-3xl font-bold text-primary-green text-center mb-6">
            Hệ thống Phát hiện Bệnh Cây bằng AI
        </h1>
        <p class="text-gray-600 text-center mb-4">
            Ứng dụng nhận diện các loại bệnh trên lá cây.
        </p>
        <h2 class="text-2xl font-semibold text-primary-green border-b border-gray-200 pb-2">
            <i class="fas fa-sign-in-alt mr-2"></i> Đăng nhập hệ thống
        </h2>
    """, unsafe_allow_html=True)


    with st.form("login_form"):
        st.markdown(f'<label for="userID" class="block text-sm font-medium text-gray-700 mb-1">ID người dùng (Demo: {USER}):</label>', unsafe_allow_html=True)
        username_input = st.text_input("", placeholder="Nhập ID", key="username_val", label_visibility="collapsed")
        
        st.markdown(f'<label for="password" class="block text-sm font-medium text-gray-700 mb-1">Mật khẩu (Demo: {PASS}):</label>', unsafe_allow_html=True)
        password_input = st.text_input("", type="password", placeholder="Nhập mật khẩu", key="password_val", label_visibility="collapsed")
        
        login_btn = st.form_submit_button("Đăng nhập")

    if login_btn:
        if username_input == USER and password_input == PASS:
            st.session_state.logged_in = True
            st.rerun() 
        else:
            st.error("Sai tài khoản hoặc mật khẩu!")

    st.markdown("</div>", unsafe_allow_html=True)
    
    render_footer()
    st.stop()

# ========================= GOOGLE DRIVE MODEL HANDLING =================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1pLZYbUXHnoIEZEHrjg2Q-bj9Q47vOKh1"
MODEL_PATH = "plant_disease_Cnn.h5"

@st.cache_resource(show_spinner=False)
def load_model_from_drive():
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
            
    with st.spinner("Đang load mô hình..."):
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model and cache it
model = load_model_from_drive()

# ========================= MAIN APP PAGE ======================
render_header()

# Main content card
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("""
    <h1 class="text-3xl font-bold text-primary-green text-center mb-6">
        🌿 Hệ thống Phát hiện Bệnh Cây
    </h1>
    <p class="text-gray-600 text-center mb-6">
        Tải lên hình ảnh lá cây để nhận diện các loại bệnh hoặc xác định lá khỏe mạnh.
    </p>
""", unsafe_allow_html=True)

# ========================= CLASS LABELS (EXPLICIT 2-CLASS ASSIGNMENT) =====================
num_classes = model.output_shape[-1]

if num_classes == 2:
    # Gán tên lớp tường minh theo yêu cầu. 
    classes = ["BỆNH", "KHỎE MẠNH"] 
else:
    # Dùng tên chung nếu có nhiều hơn 2 lớp 
    classes = [f"Class_{i}" for i in range(num_classes)]
    st.warning(f"⚠️ Mô hình có {num_classes} lớp. Vui lòng kiểm tra và thay thế danh sách classes trong code để tên bệnh hiển thị chính xác.")


# ========================= IMAGE UPLOAD & PREDICTION ======================
st.subheader("📸 Tải ảnh lá cây để nhận diện bệnh")

uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

def prepare(img):
    """Tiền xử lý ảnh: thay đổi kích thước, chuẩn hóa, thêm chiều batch."""
    img = img.resize((224, 224)) 
    img = np.asarray(img) / 255.0 
    if len(img.shape) == 2: 
        img = np.stack((img,) * 3, axis=-1)
    return np.expand_dims(img, axis=0) 

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Display image centered
    col_img_1, col_img_2, col_img_3 = st.columns([1, 2, 1])
    with col_img_2:
        st.image(img, caption="Ảnh đã tải lên", use_column_width="auto", width=300)

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
        result_name = classes[class_id].upper()

        if confidence * 100 > 70:
            st.balloons()
            # Hiển thị kết quả rõ ràng (BỆNH hoặc KHỎE MẠNH)
            st.markdown(f'<div class="stSuccess">🌿 **Kết quả Dự đoán:** <strong style="font-size: 1.25em;">{result_name}</strong></div>', unsafe_allow_html=True)
            st.metric(label="Độ tin cậy", value=f"{confidence * 100:.2f}%")
        else:
            # Hiển thị kết quả cảnh báo
            st.markdown(f'<div class="stWarning">⚠️ **Kết quả Không Rõ Ràng:** Mô hình dự đoán là <strong>{result_name}</strong> với độ tin cậy {confidence * 100:.2f}%. Vui lòng thử ảnh khác.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) # End main-card

render_footer()
