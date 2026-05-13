# AI Pulmonary Diagnostic Suite

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ai-pulmonary-diagnostic-suite-sxvewn5lvevhh7ehe22v5p.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training_Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/hoadao833hung/pneumonia-diagnosis-web)

## Overview
The AI Pulmonary Diagnostic Suite is a web-based diagnostic support system designed to assist in detecting pneumonia from chest X-rays. Built with a dual-engine architecture, the system balances high-speed batch processing with interpretable predictions, aiming to provide a reliable and explainable tool for medical image analysis.

**Live Demo:** [Web Interface Access](https://ai-pulmonary-diagnostic-suite-sxvewn5lvevhh7ehe22v5p.streamlit.app/)

## Key Features & Capabilities
- **Dual-Engine Architecture:**
    - **Rapid Triage:** Utilizes an optimized TFLite model for low-latency batch processing (supporting up to 10 simultaneous scans).
    - **Explainable AI (XAI):** Employs a full Keras model to generate Grad-CAM heatmaps, providing visual interpretation of lung opacities to build clinical trust.
- **Workflow Integration (UI/UX):**
    - Automated risk categorization: Routes scans into *Critical*, *Review Required*, or *Cleared* queues based on customizable confidence thresholds.
    - Built-in radiograph illumination controls for browser-based image enhancement.
    - Human-in-the-loop mechanism allowing end-users to validate or override AI predictions.
- **Data Export:** Batch diagnostic results can be instantly exported to Excel (.xlsx) or Clinical Text (.txt) for external record-keeping.

## Model Architecture & Performance
- **Base Model:** DenseNet121 integrated with Transfer Learning techniques.
- **Training Strategy:** Fine-tuned on a dataset of over 5,800 images utilizing extensive data augmentation. 
- **Key Metric:** Achieves a **92% Recall**, strategically optimized to minimize false negatives in potential critical cases.

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hung833/Pneumonia-Diagnosis.git](https://github.com/Hung833/Pneumonia-Diagnosis.git)
   cd Pneumonia-Diagnosis
2. **Install dependencies:**
   
   Make sure you have streamlit, tensorflow, pandas, Pillow, matplotlib, and xlsxwriter installed.
   ```bash
   pip install -r requirements.txt
3. **Download Model Weights:**

    Place the following pre-trained models into the root directory:
    - pneumonia_model.tflite (for Rapid Triage)
    - pneumonia_model_finetuned.keras (for Grad-CAM)
4. **Run the Streamlit Web App:**

   ```bash
   streamlit run app.py
## Development Roadmap
- [x] Build Baseline Model using DenseNet121.
- [x] Deploy end-to-end web application via Streamlit.
- [x] Integrate Grad-CAM for model explainability.
- [x] Convert weights to TFLite for reduced inference latency.
- [ ] Implement user authentication for secure clinical access.
- [ ] Containerize the application using Docker for easier deployment.
