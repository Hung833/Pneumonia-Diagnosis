# Pneumonia Diagnosis Web System (PDS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Giới thiệu dự án
Hệ thống chẩn đoán viêm phổi (Pneumonia Diagnosis System) là một ứng dụng AI hỗ trợ bác sĩ và nhân viên y tế trong việc phân tích ảnh X-quang phổi. Dự án sử dụng mô hình Deep Learning tiên tiến để phân loại ảnh X-quang thành hai nhóm: **Bình thường (Normal)** và **Viêm phổi (Pneumonia)**.

Dự án này tập trung vào tính chính xác và khả năng giải thích của mô hình trong y tế.

## Tính năng chính
- **AI Chẩn đoán:** Phân loại ảnh X-quang với độ chính xác cao.
- **Tối ưu Recall:** Giảm thiểu tối đa tỷ lệ bỏ sót ca bệnh (False Negative).
- **Trực quan hóa (Đang phát triển):** Sử dụng Grad-CAM để đánh dấu vùng tổn thương trên phổi.
- **Web Interface:** Giao diện người dùng thân thiện, cho phép tải ảnh và nhận kết quả tức thì.

## Tập dữ liệu
Dự án sử dụng tập dữ liệu **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)** từ Kaggle.
- **Tổng số ảnh:** 5,863 ảnh X-quang ngực (JPEG).
- **Phân loại:** Normal và Pneumonia (Viral/Bacterial).

## Kiến trúc mô hình (AI Core)
Mô hình được xây dựng dựa trên phương pháp **Transfer Learning**:
- **Base Model:** `DenseNet121` (Pre-trained trên ImageNet).
- **Cải tiến:** - Loại bỏ các lớp Fully Connected cuối cùng.
    - Thêm lớp `GlobalAveragePooling2D` để giảm tham số và tránh Overfitting.
    - Lớp `Dropout (0.5)` để tăng tính tổng quát hóa.
    - Lớp `Dense` cuối với hàm kích hoạt `Sigmoid` cho phân loại nhị phân.
- **Tiền xử lý:** Data Augmentation (Rotation, Zoom, Horizontal Flip) để giải quyết vấn đề mất cân bằng dữ liệu.

## Kết quả đạt được
Dựa trên thử nghiệm thực tế, mô hình đạt được các chỉ số ấn tượng:
- **Accuracy:** > 90%
- **Recall (Pneumonia):** Rất cao (Đảm bảo không bỏ sót bệnh nhân).
- **F1-Score:** Cân bằng tốt giữa Precision và Recall.

## 🛠 Cài đặt và Sử dụng

1. **Clone repository:**
   ```bash
   git clone [https://github.com/Hung833/Pneumonia-Diagnosis.git]
   cd pneumonia-diagnosis-web

2. **Cài đặt môi trường:**
    ```bash
    pip install -r requirements.txt

3. **Chạy ứng dụng Web:** 
    ```bash
    python app.py

## Lộ trình phát triển (Roadmap)
- Xây dựng mô hình Baseline với DenseNet121.
- Tối ưu hóa mô hình bằng Fine-tuning các block cuối.
- Tích hợp Grad-CAM để giải thích mô hình (Explainable AI).
- Chuyển đổi sang TFLite để tối ưu hóa tốc độ trên Web.
- Đóng gói ứng dụng với Docker.
