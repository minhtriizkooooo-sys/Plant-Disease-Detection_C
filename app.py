import requests
import tensorflow as tf
import streamlit as st
import os
import numpy as np
from PIL import Image

MODEL_URL = "https://raw.githubusercontent.com/minhtriizkooooo-sys/Plant-Disease-Detection_C/main/model/plant_disease_mobilenet.h5"
MODEL_PATH = "plant_disease_mobilenet.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Đang tải mô hình từ GitHub..."):
            r = requests.get(MODEL_URL)
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
            else:
                st.error(f"❌ Không thể tải mô hình từ GitHub. Mã lỗi: {r.status_code}")
                st.stop()
    with st.spinner("Đang load mô hình..."):
        # Load mô hình từ file đã tải xuống
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Thêm GlobalAveragePooling2D nếu mô hình chưa có
        if not isinstance(model.layers[-2], tf.keras.layers.GlobalAveragePooling2D):
            model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        return model

# Load mô hình
model = load_model()

# ========================= CLASS LABELS ==========================
classes = ["BỆNH", "KHỎE MẠNH"]

# ========================= PREPROCESS ==========================
def prepare(img):
    img = img.convert("RGB").resize((224, 224))  # Resize ảnh về kích thước 224x224
    x = np.array(img, dtype=np.float32)  # Chuyển ảnh thành numpy array
    x = x[:, :, ::-1]  # Chuyển từ RGB sang BGR (nếu sử dụng MobileNetV2)
    x /= 255.0  # Chuẩn hóa dữ liệu
    return np.expand_dims(x, axis=0)  # Mở rộng kích thước ảnh để phù hợp với mô hình

# ========================= MAIN UI ==========================
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

            # Dự đoán lớp cho ảnh
            pred = model.predict(x)[0]  # Dự đoán lớp
            conf = float(np.max(pred))  # Lấy độ tin cậy cao nhất
            label = classes[np.argmax(pred)]  # Lựa chọn lớp có độ tin cậy cao nhất

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
