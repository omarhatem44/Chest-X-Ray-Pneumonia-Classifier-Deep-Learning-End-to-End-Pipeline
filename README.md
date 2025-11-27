# 🧠 Chest X-Ray Pneumonia Classifier  
## ⚡ Deep Learning End-to-End Pipeline (Custom CNN)

A complete **medical imaging deep learning project** that detects **Pneumonia** vs **Normal** from **Chest X-ray images** using a custom-designed **Convolutional Neural Network (CNN)**.  
The project demonstrates strong ML engineering practices: preprocessing, augmentation, class balancing, modular scripts, visualization, and evaluation.

---

# ⭐ Key Features (Important)
- **Full ML pipeline (EDA → Preprocessing → Training → Evaluation)**  
- **Class imbalance handling** (Augmentation ONLY for NORMAL)  
- **Custom CNN baseline model**  
- **Confusion Matrix + Precision/Recall/F1-score**  
- **Production-style project structure**  
- **Clear, modular, documented code**  

---

# 📊 **Model Performance (Important)**

| Metric | Value |
|--------|--------|
| 🎯 **Test Accuracy** | **85.7%** |
| NORMAL – Precision | 0.75 |
| NORMAL – Recall | 0.92 |
| PNEUMONIA – Precision | 0.94 |
| PNEUMONIA – Recall | 0.81 |

### **Confusion Matrix**
[[216, 18],
[ 71, 319]]




---

# 📁 **Project Structure (Important)**
pneumonia-xray-classifier/
│
├── src/ # All training & evaluation scripts
│ ├── train_cnn_pneumonia.py
│ ├── evaluate_cnn_metrics.py
│ ├── augmentation_normal_only.py
│ ├── eda_pneumonia.py
│
├── results/ # Model evaluation outputs
│ └── confusion_matrix_cnn.png
│
├── models/ # (Empty – weights not uploaded)
│
├── requirements.txt
└── README.md


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

This script:

Loads dataset

Applies augmentation to NORMAL

Trains CNN

Saves best model as:

cnn_pneumonia_best.h5 (stored locally only)
---

# 📈 Evaluate the Model
```bash
python src/evaluate_cnn_metrics.py
```
Outputs:

Confusion Matrix

Precision / Recall / F1-score

Saved under results/

---
#🚀 Future Enhancements (Important)
 ResNet50 Transfer Learning

EfficientNet / DenseNet versions

Grad-CAM Explainability

Deploy via Flask / FastAPI

Convert to TFLite for mobile deployment
---
#👨‍💻 Author
Mohamed Ellabban
Machine Learning Engineer — Deep Learning & Medical AI

GitHub: https://github.com/omarhatem44

Email: mohamed.ellabban@outlook.com
---





