# skin-cancer-detection-localization
"Multi-Class Skin Cancer Detection and Localization Using Deep Learning"

# Project Overview

Skin cancer is one of the most common types of cancer globally, and early detection is crucial for effective treatment.  
This project aims to develop an **automated deep learning-based system** capable of both **classifying** and **localizing** different types of skin lesions from medical images.  
By integrating **CNN architectures (ResNet, MobileNet, EfficientNet)** with **object detection models (YOLO, Faster R-CNN)**, the system enhances diagnostic precision and provides visual localization of cancerous regions.

# Project Goals

- Detect and classify multiple types of skin cancer (Melanoma, BCC, SCC, Benign).
- Localize lesions on skin images using object detection.
- Improve interpretability using visualization tools like **Grad-CAM**.
- Create a user-friendly GUI for image upload and result display.

# Functional Requirements
1. **Image Preprocessing**
   - Noise removal, normalization, and contrast enhancement using OpenCV.
2. **Multi-Class Classification**
   - Predict lesion type using CNNs (ResNet, MobileNet, EfficientNet).
3. **Lesion Localization**
   - Apply **YOLOv5** or **Faster R-CNN** for bounding box detection.
4. **Model Evaluation**
   - Use metrics such as Accuracy, Precision, Recall, F1-score, and IoU.
5. **Visualization**
   - Apply **Grad-CAM** to interpret model predictions.
6. **User Interface**
   - Develop a GUI (e.g., with Tkinter or Streamlit) for image uploads and classification.

# Tools and Technologies
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras, PyTorch (optional for YOLO) |
| **Image Processing** | OpenCV, Pillow |
| **Machine Learning Utilities** | Scikit-learn, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook, Google Colab |
| **GUI (Optional)** | Streamlit or Tkinter |

# Dataset

### 1️⃣ **HAM10000 Dataset (Recommended)**
**📂 Kaggle Link:**  
[HAM10000 ("Human Against Machine") Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

# Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- IoU (Intersection over Union) for localization tasks  

# Model Architecture Summary
| Task | Model | Description |
|------|--------|-------------|
| Classification | ResNet50 / MobileNetV2 / EfficientNetB0 | Feature extraction and lesion classification |
| Localization | YOLOv5 / Faster R-CNN | Bounding box detection of affected regions |
| Visualization | Grad-CAM | Interpretive heatmap overlay on lesion image |

# Expected Output
- **Classification Result:** Identified lesion type  
- **Localization Output:** Bounding box highlighting affected area  
- **Grad-CAM Visualization:** Heatmap showing regions influencing the prediction  
- **GUI Interface:** Simple upload → prediction → visualization workflow  
