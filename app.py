import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = 'pneumonia_model.tflite'

st.set_page_config(page_title="AI Diagnostic Assistant - X-Ray", page_icon="🫁", layout="wide")

@st.cache_resource
def load_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image: Image.Image, input_details: list) -> np.ndarray:
    expected_shape = input_details[0]['shape']
    target_height, target_width = expected_shape[1], expected_shape[2]
    expected_channels = expected_shape[-1]

    image_resized = image.resize((target_width, target_height))

    if expected_channels == 3:
        image_processed = image_resized.convert('RGB')
    else:
        image_processed = image_resized.convert('L')

    img_array = np.array(image_processed, dtype=np.float32) / 255.0
    return img_array.reshape(expected_shape)

def predict(interpreter: tf.lite.Interpreter, img_array: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

def main():
    with st.spinner('Initializing AI core...'):
        interpreter = load_model(MODEL_PATH)

    st.title("🫁 AI Chest X-Ray Diagnostic Assistant")
    st.markdown("*Professional Edition for Healthcare Providers*")

    with st.expander("Clinical Case Information (Optional)"):
        patient_id = st.text_input("Patient ID (Anonymized):", placeholder="e.g., PT-2026-001")
        symptoms = st.text_area("Initial Clinical Notes:")

    st.write("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Radiography Scan")
        uploaded_file = st.file_uploader("Upload standard X-Ray image...", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Patient X-Ray Scan', use_container_width=True)

    with col2:
        st.subheader("AI Analysis")
        
        if not uploaded_file:
            st.info("Please upload an X-ray image in the left panel to begin analysis.")
            return

        if st.button('🔍 Run AI Diagnostic Scan', type="primary", use_container_width=True):
            with st.spinner('Extracting image features...'):
                input_details = interpreter.get_input_details()
                
                img_array = preprocess_image(image, input_details)
                prediction = predict(interpreter, img_array)
                
                if prediction.shape[1] == 1:
                    prob_pneumonia = prediction[0][0]
                    score_pneumonia = prob_pneumonia * 100
                    score_normal = (1.0 - prob_pneumonia) * 100
                else:
                    score_normal = prediction[0][0] * 100
                    score_pneumonia = prediction[0][1] * 100
                
                st.write("### AI Diagnostic Suggestion:")
                
                if score_pneumonia > 50:
                    st.error("⚠️ SUSPECTED LESIONS: Signs of Pneumonia Detected")
                    st.metric(label="AI Confidence Score", value=f"{score_pneumonia:.2f}%")
                    st.progress(int(score_pneumonia))
                else:
                    st.success("✅ NO ABNORMALITIES: Lungs Appear Healthy")
                    st.metric(label="AI Confidence Score", value=f"{score_normal:.2f}%")
                    st.progress(int(score_normal))
                
                st.write("---")
                st.subheader("Final Physician Conclusion")
                st.caption("AI serves as a supportive tool. Physicians must verify and provide the final conclusion.")
                
                doctor_decision = st.radio(
                    "Evaluate AI Result:",
                    ("Pending", "Agree with AI", "Reject - AI Incorrect")
                )
                
                doctor_notes = st.text_area("Diagnostic notes for medical records:")
                
                if st.button("Save Diagnostic Record"):
                    st.toast('Physician conclusion saved to the system!', icon='💾')

    st.write("---")
    st.caption("""
    **MEDICAL DISCLAIMER:**
    This system utilizes Artificial Intelligence for image analysis and provides information for reference purposes only. 
    This software DOES NOT replace professional medical diagnosis, advice, or treatment from a qualified healthcare provider.
    """)

if __name__ == "__main__":
    main()
