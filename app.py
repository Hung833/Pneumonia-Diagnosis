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
    """
    Hàm tự động tìm layer trích xuất đặc trưng cuối cùng dựa trên Output Shape.
    Hỗ trợ hoàn hảo cho các model Transfer Learning (MobileNet, ResNet, VGG...)
    """
    for layer in reversed(model.layers):
        try:
            # Lấy shape đầu ra của layer
            out_shape = layer.output_shape
            
            # Xử lý trường hợp layer trả về nhiều output (List)
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
                
            # Kiểm tra xem có phải là mảng 4 chiều (Batch, Height, Width, Channels) không
            # Và đảm bảo Height/Width lớn hơn 1 (Không phải là layer đã Flatten)
            if len(out_shape) == 4 and out_shape[1] is not None and out_shape[1] > 1:
                return layer.name
        except Exception:
            continue
            
    raise ValueError("Lỗi: Không tìm thấy layer nào xuất ra mảng 4D (Feature Map) trong model của bạn!")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # --- BẢN VÁ LỖI TUPLE/LIST ---
        # Nếu model trả về list, lấy phần tử đầu tiên (Tensor)
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]
        # -----------------------------
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Bắt lỗi nếu layer không có đạo hàm kết nối tới output
    if grads is None:
        raise ValueError(f"Không thể tính đạo hàm. Layer '{last_conv_layer_name}' có thể không hợp lệ.")
        
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
st.title("🩻 Hệ Thống Phân Loại & Giải Thích X-Quang Phổi")

# --- CODE DEBUG (CHÈN TẠM THỜI ĐỂ TÌM TÊN LAYER) ---
with st.expander("🛠️ Xem cấu trúc Keras Model (Chỉ dành cho Kỹ sư)"):
    debug_model = load_keras_model()
    for layer in debug_model.layers:
        try:
            st.code(f"Tên: {layer.name}  |  Shape: {layer.output_shape}")
        except Exception:
            st.code(f"Tên: {layer.name}  |  Shape: Không xác định")
# ---------------------------------------------------

# KHỞI TẠO SESSION STATE ĐỂ GIỮ DỮ LIỆU KHÔNG BỊ MẤT KHI BẤM NÚT
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
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
                                    # CHỐT ĐÍCH DANH TÊN LAYER CỦA DENSENET
                                    LAST_CONV_LAYER_NAME = 'relu'
                                    
                                    heatmap = make_gradcam_heatmap(row['img_array_for_keras'], keras_model, LAST_CONV_LAYER_NAME, pred_index=row['class_idx'])
                                    overlay_img = overlay_heatmap(row['image'], heatmap, alpha=0.5)
                                    st.image(overlay_img, caption="Vùng màu Đỏ/Vàng là nơi AI tập trung chẩn đoán", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Lỗi tạo Grad-CAM: {e}")

    with tab1: display_grid(pn_cases)
    with tab2: display_grid(rv_cases)
    with tab3: display_grid(cl_cases)

    # ==========================================
    # 5. XUẤT BÁO CÁO (THỰC THI THẬT)
    # ==========================================
    st.write("---")
    st.subheader("📥 Xuất Báo Cáo")
    colA, colB = st.columns(2)
    
    # Chuẩn bị dữ liệu thống kê
    df_export = df[['filename', 'status', 'confidence']].copy()
    df_export.columns = ['Tên file', 'Chẩn đoán AI', 'Độ tin cậy (%)']
    
    # 1. Tạo file Excel lưu vào bộ nhớ đệm
    import io
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='ThongKe')
    
    with colB:
        st.download_button(
            label="📊 Tải file Excel (Thống kê quản lý)", 
            data=excel_buffer.getvalue(), 
            file_name="BaoCao_ThongKe.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    # 2. Tạo text Báo cáo lâm sàng
    with colA:
        clinical_text = "BÁO CÁO LÂM SÀNG - HỆ THỐNG AI\n" + "="*40 + "\n"
        for _, row in df_export.iterrows():
            clinical_text += f"- File X-Quang: {row['Tên file']}\n"
            clinical_text += f"  + AI Đánh giá: {row['Chẩn đoán AI']}\n"
            clinical_text += f"  + Độ tin cậy: {row['Độ tin cậy (%)']:.2f}%\n"
            clinical_text += "-"*40 + "\n"
            
        st.download_button(
            label="📝 Tải Báo cáo Lâm sàng (Text)", 
            data=clinical_text.encode('utf-8-sig'), 
            file_name="BaoCao_LamSang.txt",
            mime="text/plain"
        )
