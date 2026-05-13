import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance
import matplotlib.cm as cm
import io

# 1. APPLICATION CONFIGURATION
st.set_page_config(page_title="AI Pulmonary Diagnostic Suite", page_icon="🫁", layout="wide")

# ==========================================
# 2. HYBRID MODEL INFERENCE ENGINE
# ==========================================
@st.cache_resource
def load_triage_model():
    """Loads lightweight TFLite model for rapid batch triage."""
    interpreter = tf.lite.Interpreter(model_path='pneumonia_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_explainable_model():
    """Loads full Keras model for Grad-CAM heatmaps."""
    return tf.keras.models.load_model('pneumonia_model_finetuned.keras')

triage_interpreter = load_triage_model()

# ==========================================
# 3. CORE AI & UTILITY FUNCTIONS
# ==========================================
def run_rapid_triage(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def generate_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]
        
        if preds.shape[1] == 1:
            pred_index = 0
        elif pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        raise ValueError(f"Gradient computation failed for layer: '{last_conv_layer_name}'.")
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap_overlay(img_pil, heatmap, alpha=0.4):
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
# 4. CLINICAL WORKSPACE UI
# ==========================================
st.title("🫁 Pulmonary X-Ray Triage & Analytics Suite")
st.markdown("*Advanced Clinical Decision Support System*")

# State Management Initialization
if 'clinical_records' not in st.session_state:
    st.session_state.clinical_records = []

with st.sidebar:
    st.header("📁 Patient Scan Import")
    uploaded_files = st.file_uploader(
        "Upload standard DICOM/JPEG/PNG scans (Max 10)", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Batch limit exceeded. Processing the first 10 scans.")
            uploaded_files = uploaded_files[:10]
            
        if st.button('🔍 Execute Batch Triage', type="primary", use_container_width=True):
            with st.spinner("AI engine analyzing patient scans..."):
                st.session_state.clinical_records = [] 
                
                for file in uploaded_files:
                    image = Image.open(file)
                    input_details = triage_interpreter.get_input_details()
                    expected_shape = input_details[0]['shape']
                    expected_channels = expected_shape[-1]
                    
                    img_resized = image.resize((expected_shape[1], expected_shape[2]))
                    img_processed = img_resized.convert('RGB' if expected_channels == 3 else 'L')
                    img_array = np.array(img_processed) / 255.0
                    img_array = img_array.reshape(expected_shape)
                    
                    prediction = run_rapid_triage(triage_interpreter, img_array)
                    prob_pneumonia = float(prediction[0][0]) if prediction.shape[1] == 1 else float(prediction[0][1])
                    prob_normal = 1.0 - prob_pneumonia
                    
                    status = "Pneumonia" if prob_pneumonia > 0.5 else "Normal"
                    confidence = prob_pneumonia if status == "Pneumonia" else prob_normal
                    
                    st.session_state.clinical_records.append({
                        "filename": file.name,
                        "image": image, 
                        "tensor_array": img_array, 
                        "status": status,
                        "confidence": confidence * 100,
                        "class_idx": 1 if status == "Pneumonia" else 0
                    })

# DATA PRESENTATION LAYER
if st.session_state.clinical_records:
    df = pd.DataFrame(st.session_state.clinical_records)
    
    critical_cases = df[df['status'] == "Pneumonia"].sort_values(by="confidence", ascending=False)
    review_cases = df[(df['status'] == "Normal") & (df['confidence'] < 85)].sort_values(by="confidence", ascending=True)
    cleared_cases = df[(df['status'] == "Normal") & (df['confidence'] >= 85)].sort_values(by="confidence", ascending=False)

    st.write("---")
    tab1, tab2, tab3 = st.tabs([
        f"🚨 Critical: Suspected Pneumonia ({len(critical_cases)})", 
        f"⚠️ Review Required ({len(review_cases)})", 
        f"✅ Cleared: Normal ({len(cleared_cases)})"
    ])

    def render_patient_grid(cases):
        if cases.empty:
            st.info("No patient records in this triage category.")
            return
            
        cols = st.columns(2)
        for i, (_, row) in enumerate(cases.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    # 1. Radiograph Enhancement
                    brightness = st.slider("Radiograph Illumination", 0.5, 2.0, 1.0, key=f"br_{row['filename']}")
                    enhancer = ImageEnhance.Brightness(row['image'])
                    enhanced_im = enhancer.enhance(brightness)
                    
                    st.image(enhanced_im, use_container_width=True)
                    st.markdown(f"**Scan ID:** `{row['filename']}` | **AI Confidence:** `{row['confidence']:.2f}%`")
                    
                    # 2. Medical Verification
                    if row['status'] == "Normal" and row['confidence'] < 80:
                        st.error("❗ Clinical Alert: Low diagnostic confidence. Secondary physician review mandated.")
                    
                    st.radio(
                        "Physician Assessment:", 
                        ["Pending Review", "Concur with AI", "Dissent (Override AI)"], 
                        horizontal=True, 
                        key=f"eval_{row['filename']}"
                    )
                    
                    # 3. Explainable AI On-Demand
                    with st.expander("🔬 Request Explainable AI (Grad-CAM)"):
                        if st.button("Generate Lesion Localization Map", key=f"cam_{row['filename']}"):
                            with st.spinner("Initializing Deep Diagnostic Engine..."):
                                try:
                                    xai_model = load_explainable_model()
                                    LAST_CONV_LAYER_NAME = 'relu' # Adjust based on your DenseNet architecture
                                    
                                    heatmap = generate_gradcam_heatmap(row['tensor_array'], xai_model, LAST_CONV_LAYER_NAME, pred_index=row['class_idx'])
                                    overlay_img = apply_heatmap_overlay(row['image'], heatmap, alpha=0.5)
                                    st.image(overlay_img, caption="Thermal regions indicate primary focal points of AI analysis.", use_container_width=True)
                                except Exception as e:
                                    st.error(f"XAI Engine Error: {e}")

    with tab1: render_patient_grid(critical_cases)
    with tab2: render_patient_grid(review_cases)
    with tab3: render_patient_grid(cleared_cases)

    # ==========================================
    # 5. CLINICAL REPORT EXPORT
    # ==========================================
    st.write("---")
    st.subheader("📥 Export Clinical Records")
    col_txt, col_xls = st.columns(2)
    
    df_export = df[['filename', 'status', 'confidence']].copy()
    df_export.columns = ['Scan ID', 'AI Diagnosis', 'Confidence Score (%)']
    
    # Excel Export
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Triage_Data')
    
    with col_xls:
        st.download_button(
            label="📊 Download Batch Analytics (Excel)", 
            data=excel_buffer.getvalue(), 
            file_name="Department_Triage_Stats.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    # Text Export
    with col_txt:
        clinical_text = "AI TRIAGE CLINICAL REPORT\n" + "="*45 + "\n"
        for _, row in df_export.iterrows():
            clinical_text += f"Scan Reference: {row['Scan ID']}\n"
            clinical_text += f"  > Diagnostic Suggestion: {row['AI Diagnosis']}\n"
            clinical_text += f"  > System Confidence: {row['Confidence Score (%)']:.2f}%\n"
            clinical_text += "-"*45 + "\n"
            
        st.download_button(
            label="📝 Download Clinical Report (TXT)", 
            data=clinical_text.encode('utf-8-sig'), 
            file_name="Clinical_Report_Batch.txt",
            mime="text/plain",
            use_container_width=True
        )
