# AI Pulmonary Diagnostic Suite

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://pneumonia-diagnosis-24ob8csljfy2efc6zclrbp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Training_Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/hoadao833hung/pneumonia-diagnosis-web)

## Overview
The **AI Pulmonary Diagnostic Suite** is a professional-grade medical imaging tool designed to assist radiologists in detecting Pneumonia from Chest X-rays. Unlike standard classification models, this system provides a dual-layer approach: **Rapid Triage** for speed and **Explainable AI** for clinical trust.

**Live Demo:** [Click here to access the Web Interface](https://pneumonia-diagnosis-24ob8csljfy2efc6zclrbp.streamlit.app/)

## Key Features
- **Hybrid Inference Engine:**
    - **Rapid Triage:** Uses an optimized **TFLite** model for lightning-fast batch processing.
    - **Explainable Analysis (XAI):** Generates **Grad-CAM heatmaps** to highlight specific lung opacities detected by the AI.
- **Clinical Reporting:** Export diagnostic results directly to **Excel (.xlsx)** or clinical text summaries for medical records using Pandas and XlsxWriter.
- **High Sensitivity (Recall: 92%):** Specifically tuned to ensure no potential cases are missed (minimizing false negatives).

## Model & Training
- **Architecture:** DenseNet121 with Transfer Learning.
- **Optimization:** Fine-tuned on a dataset of 5,800+ images with heavy Data Augmentation. Converted to `.tflite` to reduce model size and latency.

## Local Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/Hung833/Pneumonia-Diagnosis.git](https://github.com/Hung833/Pneumonia-Diagnosis.git)
   cd Pneumonia-Diagnosis
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit Web App:**
   ```bash
   streamlit run app.py
## Development Roadmap
- Build Baseline Model using DenseNet121.
- Deploy end-to-end web application via Streamlit.
- Optimize model via Fine-tuning the top convolutional blocks.
- Integrate Grad-CAM for model explainability.
- Convert weights to TFLite to further reduce inference latency.
- Containerize the application using Docker.
