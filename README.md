# 🧠 Chest X-Ray Pneumonia Classifier (Deep Learning)
### ⚡ Deep Learning End-to-End Pipeline (Custom CNN)

A complete **medical imaging deep learning project** for binary classification (**Pneumonia vs Normal**) using a custom-designed **Convolutional Neural Network (CNN)**.

The project demonstrates strong ML engineering practices:  
➡️ Preprocessing  
➡️ Data augmentation  
➡️ Class balancing  
➡️ Modular training scripts  
➡️ Visualization + Evaluation metrics  

---

## ⭐ Key Features (Important)
- **Full ML pipeline** (EDA → Preprocessing → Training → Evaluation)  
- **Handles class imbalance** (Augmentation ONLY for NORMAL)  
- **Custom CNN baseline model**  
- **Confusion Matrix + Precision/Recall/F1-score**  
- **Production-style project structure**  
- **Clean, modular, documented code**

---

## 📊 Model Performance (Important)

| Metric | Value |
|--------|--------|
| **Test Accuracy** | **85.7%** |
| NORMAL – Precision | 0.75 |
| NORMAL – Recall | 0.92 |
| PNEUMONIA – Precision | 0.94 |
| PNEUMONIA – Recall | 0.81 |

### 🔍 Confusion Matrix  
|      | Pred Normal | Pred Pneumonia |
|------|-------------|----------------|
| **Actual Normal** | 216 | 18 |
| **Actual Pneumonia** | 71 | 319 |

---
### 📉 Confusion Matrix (Visualization)

<p align="center">
  <img src="results/confusion_matrix_cnn.png" width="450">
</p>

## 📁 Project Structure (Important)
```bash
pneumonia-xray-classifier/
│
├── src/                       # All training & evaluation scripts
│   ├── train_cnn_pneumonia.py
│   ├── evaluate_cnn_metrics.py
│   ├── augmentation_normal_only.py
│   ├── eda_pneumonia.py
│
├── results/                   # Model evaluation outputs
│   └── confusion_matrix_cnn.png
│
├── models/                    # (Empty – weights not uploaded)
│
└── README.md

```

---

# ⚙️ Installation

```bash
pip install -r requirements.txt
```
---

# 🏋️‍♂️ Train the Model
```bash
python src/train_cnn_pneumonia.py
```

### **This script:**

- Loads dataset

- Applies augmentation to NORMAL

- Trains CNN

- Saves best model as:

cnn_pneumonia_best.h5 (stored locally only)
---

# 📈 Evaluate the Model
```bash
python src/evaluate_cnn_metrics.py
```
### **Outputs:**

- Confusion Matrix

- Precision / Recall / F1-score

- Saved under results/

---
# 🚀 Future Enhancements (Important)
     
- ResNet50 Transfer Learning
    
- EfficientNet / DenseNet versions
    
- Grad-CAM Explainability
    
- Deploy via Flask / FastAPI
      
---

## 👨‍💻 Author  
**Omar Hatem Ellaban**  
Machine Learning & Deep Learning Engineer  

📧 Email: **omarhatemmoahemd@gmail.com**  

---





