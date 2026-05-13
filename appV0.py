import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image # Thư viện xử lý ảnh

# 1. TIÊU ĐỀ VÀ GIAO DIỆN CƠ BẢN
st.set_page_config(page_title="AI Bác Sĩ - Chẩn Đoán Phổi", page_icon="🩺")
st.title("🩺 Hệ Thống Chẩn Đoán Viêm Phổi AI")
st.write("Vui lòng upload ảnh X-quang phổi để hệ thống phân tích.")

# 2. HÀM LOAD MODEL (Chạy 1 lần duy nhất để tiết kiệm thời gian)
# @st.cache_resource giúp lưu model vào bộ nhớ đệm, không cần load lại mỗi khi bấm nút
@st.cache_resource
def load_model():
    # Đường dẫn đến file model bạn đã tải về
    # Nếu dùng file .keras thì đổi tên bên dưới
    model = tf.keras.models.load_model('pneumonia_model_v1.keras')
    return model

# Gọi hàm để lấy model ra dùng
with st.spinner('Đang khởi động bộ não AI...'):
    model = load_model()

# 3. CHỨC NĂNG UPLOAD ẢNH
uploaded_file = st.file_uploader("Chọn ảnh X-quang (đuôi jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng vừa upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh X-quang đã tải lên', use_container_width=True)

    # 4. NÚT BẤM DỰ ĐOÁN
    if st.button('🔍 Phân tích ngay'):
        with st.spinner('AI đang soi phim...'):
            
            # --- BƯỚC QUAN TRỌNG: TIỀN XỬ LÝ (PREPROCESSING) ---
            # Phải làm Y HỆT lúc train model (Resize, Grayscale, Normalize)
            
            # a. Chuyển sang ảnh đen trắng (Grayscale - L mode)
            # Vì lúc train mình dùng cv2.imread(..., 0) nên giờ phải chuyển về đen trắng
            image_gray = image.convert('L')
            
            # b. Resize về 224x224
            image_resized = image_gray.resize((224, 224))
            
            # c. Chuyển thành mảng số (NumPy array)
            img_array = np.array(image_resized)
            
            # d. Normalize (Chia cho 255)
            img_array = img_array / 255.0
            
            # e. Reshape thành khối hộp (1, 224, 224, 1)
            # Số 1 đầu tiên là batch_size (1 tấm ảnh)
            # Số 1 cuối cùng là kênh màu (đen trắng)
            img_array = img_array.reshape(1, 224, 224, 1)
            
            # --- BƯỚC 5: DỰ ĐOÁN (PREDICT) ---
            prediction = model.predict(img_array)
            
            # Kết quả prediction sẽ là một danh sách, ví dụ: [[0.1, 0.9]]
            # Số thứ nhất (Index 0): Tỉ lệ NORMAL
            # Số thứ hai (Index 1): Tỉ lệ PNEUMONIA
            
            score_normal = prediction[0][0] * 100     # Nhân 100 để ra %
            score_pneumonia = prediction[0][1] * 100
            
            # --- BƯỚC 6: HIỂN THỊ KẾT QUẢ ---
            st.write("---")
            st.subheader("Kết quả chẩn đoán:")
            
            # Logic hiển thị
            if score_pneumonia > 50:
                st.error(f"⚠️ CẢNH BÁO: PHÁT HIỆN DẤU HIỆU VIÊM PHỔI")
                st.write(f"Độ tin cậy: **{score_pneumonia:.2f}%**")
                st.progress(int(score_pneumonia)) # Thanh tiến trình màu đỏ
            else:
                st.success(f"✅ KẾT QUẢ: PHỔI BÌNH THƯỜNG")
                st.write(f"Độ tin cậy: **{score_normal:.2f}%**")
                st.progress(int(score_normal)) # Thanh tiến trình màu xanh
            
            st.info("Lưu ý: Kết quả này chỉ mang tính tham khảo kỹ thuật, không thay thế chẩn đoán của bác sĩ.")
