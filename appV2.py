import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="AI Diagnostic Assistant - Batch Mode", page_icon="🫁", layout="wide")

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='pneumonia_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

interpreter = load_tflite_model()

# 2. GIAO DIỆN CHÍNH
st.title("🫁Hệ Thống Trợ Lý AI Phân Loại X-Quang")
st.markdown("Hỗ trợ bác sĩ xử lý nhanh danh sách bệnh nhân dựa trên mức độ ưu tiên.")

# 3. SIDEBAR: TẢI FILE
with st.sidebar:
    st.header("📁 Tải Dữ Liệu")
    uploaded_files = st.file_uploader(
        "Chọn tối đa 10 ảnh X-quang", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Bạn đã chọn quá 10 ảnh. Hệ thống sẽ chỉ xử lý 10 ảnh đầu tiên.")
            uploaded_files = uploaded_files[:10]
    
    analyze_button = st.button('🔍 Bắt đầu Phân tích Hàng loạt', type="primary", use_container_width=True)

# 4. LOGIC PHÂN TÍCH
results = []

if uploaded_files and analyze_button:
    progress_bar = st.progress(0)
    for index, file in enumerate(uploaded_files):
        image = Image.open(file)
        
        # Tiền xử lý
        input_details = interpreter.get_input_details()
        expected_shape = input_details[0]['shape']
        expected_channels = expected_shape[-1]
        
        img_resized = image.resize((expected_shape[1], expected_shape[2]))
        img_processed = img_resized.convert('RGB' if expected_channels == 3 else 'L')
        img_array = np.array(img_processed) / 255.0
        img_array = img_array.reshape(expected_shape)
        
        # Dự đoán
        prediction = predict_tflite(interpreter, img_array)
        
        if prediction.shape[1] == 1:
            prob_pneumonia = float(prediction[0][0])
        else:
            prob_pneumonia = float(prediction[0][1])
            
        prob_normal = 1.0 - prob_pneumonia
        
        # Phân loại dựa trên ngưỡng y tế
        status = "Pneumonia" if prob_pneumonia > 0.5 else "Normal"
        confidence = prob_pneumonia if status == "Pneumonia" else prob_normal
        
        results.append({
            "filename": file.name,
            "image": image,
            "status": status,
            "confidence": confidence * 100,
            "prob_p": prob_pneumonia * 100
        })
        progress_bar.progress((index + 1) / len(uploaded_files))

# 5. HIỂN THỊ KẾT QUẢ (UX DASHBOARD)
if results:
    # Chuyển thành DataFrame để sắp xếp
    df = pd.DataFrame(results)
    
    # Nhóm 1: Ca nghi ngờ Viêm phổi (Sắp xếp độ tin cậy từ cao đến thấp)
    pneumonia_cases = df[df['status'] == "Pneumonia"].sort_values(by="confidence", ascending=False)
    
    # Nhóm 2: Ca Bình thường nhưng AI không chắc chắn (Confidence < 85%) - CẦN RÀ SOÁT
    review_cases = df[(df['status'] == "Normal") & (df['confidence'] < 85)].sort_values(by="confidence", ascending=True)
    
    # Nhóm 3: Ca Bình thường rõ rệt
    clear_cases = df[(df['status'] == "Normal") & (df['confidence'] >= 85)].sort_values(by="confidence", ascending=False)

    st.write("---")
    
    # TABS để bác sĩ dễ quản lý
    tab1, tab2, tab3 = st.tabs([
        f"🚨 Nghi Viêm Phổi ({len(pneumonia_cases)})", 
        f"⚠️ Cần Rà Soát ({len(review_cases)})", 
        f"✅ Bình Thường ({len(clear_cases)})"
    ])

    def display_grid(cases):
        if cases.empty:
            st.info("Không có dữ liệu trong mục này.")
            return
        
        # Chia 2 cột để dễ nhìn (Dạng lưới)
        cols = st.columns(2)
        for i, (_, row) in enumerate(cases.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.image(row['image'], use_container_width=True)
                    st.write(f"**File:** {row['filename']}")
                    st.write(f"**Độ tin cậy AI:** {row['confidence']:.2f}%")
                    # Nút để bác sĩ phản hồi nhanh
                    st.radio(f"Kết luận bác sĩ ({row['filename']}):", ["Chưa duyệt", "Đồng ý", "Sai"], horizontal=True, key=row['filename'])

    with tab1:
        display_grid(pneumonia_cases)
    with tab2:
        st.warning("Đây là những ca AI dự đoán là Bình Thường nhưng mức độ tin cậy thấp. Bác sĩ nên kiểm tra kỹ để tránh bỏ sót bệnh (False Negative).")
        display_grid(review_cases)
    with tab3:
        display_grid(clear_cases)

else:
    if not uploaded_files:
        st.info("💡 Mẹo: Bạn có thể chọn nhiều ảnh cùng lúc bằng cách giữ phím Ctrl/Command khi chọn file.")

# DISCLAIMER
st.write("---")
st.caption("MEDICAL DISCLAIMER: Thông tin chỉ mang tính chất tham khảo kỹ thuật.")
