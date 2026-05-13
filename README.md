# Pneumonia Diagnosis Web System (PDS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training_Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/hoadao833hung/pneumonia-diagnosis-web)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview
The Pneumonia Diagnosis System is an end-to-end AI-powered web application designed to assist medical professionals in analyzing Chest X-rays. By leveraging advanced Deep Learning techniques, the system classifies X-ray scans into two categories: **Normal** and **Pneumonia**, prioritizing high sensitivity to ensure patient safety.

**[View the Full Training Notebook on Kaggle](https://www.kaggle.com/code/hoadao833hung/pneumonia-diagnosis-web)**

## Key Features
- **High-Accuracy Classification:** Powered by a transfer-learning Deep Learning engine.
- **Clinical Safety First:** Optimized specifically for high **Recall** to minimize False Negatives (missed diagnoses).
- **Real-time Web Interface:** Built with Streamlit, enabling users to upload X-rays and receive diagnostic results with an inference latency of under 5 seconds.
- **Explainable AI (WIP):** Integrating Grad-CAM to highlight infected regions on the lungs.

## Dataset
The model was trained on the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle.
- **Total Images:** 5,863 Chest X-ray images (JPEG).
- **Classes:** Normal vs. Pneumonia (Viral/Bacterial).

## Model Architecture (AI Core)
The core engine utilizes **Transfer Learning** to maximize feature reuse on medical data:
- **Base Model:** `DenseNet121` (Pre-trained on ImageNet).
- **Custom Enhancements:**
  - Base layers frozen to retain broad spatial hierarchies.
  - Appended `GlobalAveragePooling2D` to reduce parameters.
  - Implemented `Dropout (0.5)` to aggressively combat overfitting on the specialized dataset.
  - Final `Dense` layer with `Sigmoid` activation for binary classification.
- **Preprocessing:** Strategic Data Augmentation (Rotation, Zoom, Horizontal Flip) via `ImageDataGenerator` to handle data imbalance.

## Performance Metrics
Based on the test set evaluation, the model achieved highly reliable clinical metrics:
- **Recall (Sensitivity): 92%** (Crucial metric: ensuring we successfully detect the vast majority of pneumonia cases).
- **Accuracy:** > 90%
- **F1-Score:** Maintains a strong balance between Precision and Recall.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hung833/Pneumonia-Diagnosis.git](https://github.com/Hung833/Pneumonia-Diagnosis.git)
   cd Pneumonia-Diagnosis
2. **Install dependencies:**
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
