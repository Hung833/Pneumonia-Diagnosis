import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 1. TIÊU ĐỀ VÀ CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP
# Sử dụng layout wide để có không gian hiển thị ảnh X-quang to và rõ hơn
st.set_page_config(page_title="AI Diagnostic Assistant - X-Ray", page_icon="🩻", layout="wide")

# 2. HÀM LOAD MODEL TFLITE
@st.cache_resource
def load_tflite_model():
    # Sử dụng Interpreter của TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path='pneumonia_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, img_array):
    # Lấy thông tin input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Ép kiểu dữ liệu cho đúng với yêu cầu của TFLite (thường là float32)
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    
    # Chạy dự đoán
    interpreter.invoke()
    
    # Lấy kết quả
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

with st.spinner('Đang tải hệ thống trợ lý AI...'):
    interpreter = load_tflite_model()

# 3. HEADER & THÔNG TIN BỆNH NHÂN (ẨN DANH)
st.title("🩻 Hệ Thống Trợ Lý AI Hỗ Trợ Chẩn Đoán X-Quang Phổi")
st.markdown("*Phiên bản Dành cho Chuyên gia Y tế*")

with st.expander("📝 Thông tin ca lâm sàng (Không bắt buộc)"):
    patient_id = st.text_input("Mã bệnh nhân (Đã ẩn danh):", placeholder="VD: BN-2026-001")
    symptoms = st.text_area("Ghi chú lâm sàng ban đầu của bác sĩ:")

st.write("---")

# 4. CHIA CỘT GIAO DIỆN: TRÁI (ẢNH) - PHẢI (PHÂN TÍCH AI & QUYẾT ĐỊNH CỦA BÁC SĨ)
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🖼️ Phim X-Quang")
    uploaded_file = st.file_uploader("Tải lên ảnh X-quang chuẩn (DICOM, JPG, PNG)...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Phim X-Quang bệnh nhân', use_container_width=True)

with col2:
    st.subheader("🤖 Trợ lý AI Phân Tích")
    
    if uploaded_file is None:
        st.info("Vui lòng tải phim X-quang ở cột bên trái để bắt đầu phân tích.")
    else:
        if st.button('🔍 Yêu cầu AI quét ảnh', type="primary", use_container_width=True):
            with st.spinner('Hệ thống đang trích xuất đặc trưng hình ảnh...'):
                
                # --- TIỀN XỬ LÝ ẢNH ---
                image_gray = image.convert('L')
                image_resized = image_gray.resize((224, 224))
                img_array = np.array(image_resized)
                img_array = img_array / 255.0
                img_array = img_array.reshape(1, 224, 224, 1)
                
                # --- DỰ ĐOÁN VỚI TFLITE ---
                prediction = predict_tflite(interpreter, img_array)
                
                score_normal = prediction[0][0] * 100
                score_pneumonia = prediction[0][1] * 100
                
                # --- HIỂN THỊ KẾT QUẢ THEO CHUẨN Y TẾ ---
                st.write("### Gợi ý từ AI:")
                
                if score_pneumonia > 50:
                    st.error("⚠️ TỔN THƯƠNG NGHI NGỜ: Có dấu hiệu Viêm Phổi")
                    st.metric(label="Mức độ tin cậy của AI", value=f"{score_pneumonia:.2f}%")
                    st.progress(int(score_pneumonia))
                else:
                    st.success("✅ KHÔNG TÌM THẤY BẤT THƯỜNG: Phổi có vẻ khỏe mạnh")
                    st.metric(label="Mức độ tin cậy của AI", value=f"{score_normal:.2f}%")
                    st.progress(int(score_normal))
                
                # --- KHUNG QUYẾT ĐỊNH CỦA BÁC SĨ (ETHICAL AI) ---
                st.write("---")
                st.subheader("👨‍⚕️ Kết luận cuối cùng của Bác sĩ")
                st.caption("AI chỉ đóng vai trò khoanh vùng hỗ trợ. Bác sĩ vui lòng đối chiếu và đưa ra kết luận.")
                
                doctor_decision = st.radio(
                    "Đánh giá kết quả của AI:",
                    ("Chưa đưa ra quyết định", "Đồng ý với AI", "Bác bỏ - Kết quả AI sai")
                )
                
                doctor_notes = st.text_area("Ghi chú chẩn đoán lưu vào hồ sơ:")
                
                if st.button("Lưu hồ sơ chẩn đoán"):
                    # Ở đây sau này bạn có thể viết code lưu vào Database (MongoDB/PostgreSQL)
                    st.toast('Đã lưu kết luận của bác sĩ vào hệ thống!', icon='💾')

# 5. DISCLAIMER ĐẠO ĐỨC (BẮT BUỘC Ở CUỐI TRANG)
st.write("---")
st.caption("""
**TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM Y TẾ (MEDICAL DISCLAIMER):**
Hệ thống này sử dụng Trí tuệ Nhân tạo để phân tích hình ảnh và chỉ cung cấp thông tin tham khảo. 
Phần mềm KHÔNG thể thay thế chẩn đoán y khoa chuyên nghiệp, lời khuyên, hoặc điều trị từ bác sĩ chuyên khoa.
""")
