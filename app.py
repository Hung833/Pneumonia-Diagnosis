import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🫁", layout="wide")

# ==========================================
# 2. HỆ THỐNG LOAD MODEL KÉP (HYBRID)
# ==========================================
@st.cache_resource
def load_tflite_model():
    # Model nhẹ dùng để triage (phân loại hàng loạt)
    interpreter = tf.lite.Interpreter(model_path='pneumonia_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_keras_model():
    # Model nặng dùng để chạy Grad-CAM (Giải thích AI)
    model = tf.keras.models.load_model('pneumonia_model_finetuned.keras')
    return model

interpreter = load_tflite_model()

# ==========================================
# 3. CÁC HÀM XỬ LÝ AI & TỰ ĐỘNG HÓA
# ==========================================
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def get_last_conv_layer_name(model):
    """Hàm tự động tìm layer Convolutional cuối cùng trong model"""
    for layer in reversed(model.layers):
        # Trường hợp dùng Transfer Learning (Model bọc trong Model)
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return inner_layer.name
        # Trường hợp model Sequential bình thường
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Không tìm thấy layer Conv2D nào trong model!")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.4):
    img_array = np.array(img_pil.convert('RGB'))
    heatmap_resized = Image.fromarray(heatmap).resize((img_array.shape[1], img_array.shape[0]))
    heatmap_resized = np.array(heatmap_resized)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(heatmap_resized * 255)]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap).resize((img_array.shape[1], img_array.shape[0]))
    superimposed_img = np.array(jet_heatmap) * alpha + img_array * (1 - alpha)
    return tf.keras.utils.array_to_img(superimposed_img)

# ==========================================
# 4. GIAO DIỆN & QUẢN LÝ STATE
# ==========================================
st.title("🫁 Hệ Thống Phân Loại & Giải Thích X-Quang Phổi")

# KHỞI TẠO SESSION STATE ĐỂ GIỮ DỮ LIỆU KHÔNG BỊ MẤT KHI BẤM NÚT
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

with st.sidebar:
    st.header("📁 Tải File Bệnh Nhân")
    uploaded_files = st.file_uploader("Chọn tối đa 10 ảnh", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Hệ thống chỉ lấy 10 ảnh đầu tiên.")
            uploaded_files = uploaded_files[:10]
            
        if st.button('🔍 Phân tích Triage Hàng Loạt', type="primary", use_container_width=True):
            with st.spinner("AI đang quét nhanh qua các ảnh..."):
                st.session_state.analysis_results = [] # Reset data cũ
                
                for file in uploaded_files:
                    image = Image.open(file)
                    
                    # Tiền xử lý cho TFLite
                    input_details = interpreter.get_input_details()
                    expected_shape = input_details[0]['shape']
                    expected_channels = expected_shape[-1]
                    
                    img_resized = image.resize((expected_shape[1], expected_shape[2]))
                    img_processed = img_resized.convert('RGB' if expected_channels == 3 else 'L')
                    img_array = np.array(img_processed) / 255.0
                    img_array = img_array.reshape(expected_shape)
                    
                    # Dự đoán TFLite
                    prediction = predict_tflite(interpreter, img_array)
                    if prediction.shape[1] == 1:
                        prob_pneumonia = float(prediction[0][0])
                    else:
                        prob_pneumonia = float(prediction[0][1])
                        
                    prob_normal = 1.0 - prob_pneumonia
                    status = "Pneumonia" if prob_pneumonia > 0.5 else "Normal"
                    confidence = prob_pneumonia if status == "Pneumonia" else prob_normal
                    
                    # LƯU VÀO STATE ĐỂ KHÔNG BỊ MẤT
                    st.session_state.analysis_results.append({
                        "filename": file.name,
                        "image": image, # Lưu lại PIL Image gốc
                        "img_array_for_keras": img_array, # Lưu mảng đã xử lý để dùng cho Keras nếu cần
                        "status": status,
                        "confidence": confidence * 100,
                        "class_idx": 1 if status == "Pneumonia" else 0
                    })

# HIỂN THỊ DỮ LIỆU TỪ SESSION STATE
if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results)
    
    pn_cases = df[df['status'] == "Pneumonia"].sort_values(by="confidence", ascending=False)
    rv_cases = df[(df['status'] == "Normal") & (df['confidence'] < 85)].sort_values(by="confidence", ascending=True)
    cl_cases = df[(df['status'] == "Normal") & (df['confidence'] >= 85)].sort_values(by="confidence", ascending=False)

    st.write("---")
    tab1, tab2, tab3 = st.tabs([f"🚨 Nghi Viêm Phổi ({len(pn_cases)})", f"⚠️ Cần Rà Soát ({len(rv_cases)})", f"✅ Bình Thường ({len(cl_cases)})"])

    def display_grid(cases):
        if cases.empty:
            st.info("Không có bệnh nhân trong nhóm này.")
            return
            
        cols = st.columns(2)
        for i, (_, row) in enumerate(cases.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    # 1. Chỉnh sáng/tương phản (Slider)
                    brightness = st.slider("Chỉnh sáng X-Quang", 0.5, 2.0, 1.0, key=f"br_{row['filename']}")
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(row['image'])
                    enhanced_im = enhancer.enhance(brightness)
                    
                    st.image(enhanced_im, use_container_width=True)
                    st.markdown(f"**File:** `{row['filename']}` | **Độ tin cậy AI:** `{row['confidence']:.2f}%`")
                    
                    # 2. XÁC NHẬN Y TẾ
                    if row['status'] == "Normal" and row['confidence'] < 80:
                        st.error("❗ Cảnh báo y khoa: AI không tự tin. Yêu cầu Bác sĩ kiểm tra chéo (Double Check).")
                    
                    st.radio("Kết luận của bác sĩ:", ["Chưa duyệt", "Đồng ý", "Từ chối (AI sai)"], horizontal=True, key=f"radio_{row['filename']}")
                    
                    # 3. KIẾN TRÚC HYBRID: GỌI KERAS KHI CẦN GIẢI THÍCH (ON-DEMAND)
                    with st.expander("🔬 Yêu cầu AI giải thích (Grad-CAM)"):
                        if st.button("Tạo bản đồ nhiệt tổn thương", key=f"btn_cam_{row['filename']}"):
                            with st.spinner("Đang tải model Keras phân tích chuyên sâu..."):
                                keras_model = load_keras_model()
                                
                                try:
                                    # GỌI HÀM TỰ ĐỘNG TÌM LAYER (Không cần hardcode nữa)
                                    auto_layer_name = get_last_conv_layer_name(keras_model)
                                    
                                    heatmap = make_gradcam_heatmap(row['img_array_for_keras'], keras_model, auto_layer_name, pred_index=row['class_idx'])
                                    overlay_img = overlay_heatmap(row['image'], heatmap, alpha=0.5)
                                    st.image(overlay_img, caption=f"AI quét qua layer: {auto_layer_name}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Lỗi tạo Grad-CAM: {e}")

    with tab1: display_grid(pn_cases)
    with tab2: display_grid(rv_cases)
    with tab3: display_grid(cl_cases)

    # NÚT XUẤT BÁO CÁO (Demo logic hiển thị)
    st.write("---")
    colA, colB = st.columns(2)
    with colA:
        st.download_button("📥 Xuất báo cáo Lâm sàng (PDF)", data="Dữ liệu giả lập PDF...", file_name="BenhAn.pdf")
    with colB:
        st.download_button("📊 Xuất báo cáo Quản lý (Excel)", data="Dữ liệu giả lập CSV...", file_name="ThongKe.csv")
