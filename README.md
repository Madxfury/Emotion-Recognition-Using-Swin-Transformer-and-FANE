# **Swin-FANE: Vision-Based Emotion Recognition for Elderly Care**

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=plastic)
![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-blue.svg?style=plastic)
![CUDA 12](https://img.shields.io/badge/cuda-12-blue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

Welcome to the official repository for **Swin-FANE**, a **vision-based emotion recognition system** designed to enhance affective understanding in **elderly care environments**.  
This project integrates the **Facial Attention Network Embedding (FANE)** with the **Swin Transformer** to achieve high accuracy in recognizing subtle facial emotions in aging populations.  

---

## 📘 **Overview**

Accurate **emotion recognition in elderly individuals** plays a crucial role in improving social well-being and care quality.  
Traditional CNN-based models often struggle to capture fine-grained expressions due to wrinkles, texture variations, and illumination challenges.  
To overcome these issues, this project introduces a **hybrid deep learning architecture** combining:

- **FANE (Facial Attention Network Embedding)** — focuses on key expression regions like eyes, mouth, and eyebrows.  
- **Swin Transformer** — captures both local and global contextual cues through hierarchical shifted window attention.

The proposed model is trained and tested on the **FANE dataset**, achieving superior accuracy and robustness compared to classical architectures.

---

## 🧠 **Key Features**

- **Hybrid Attention-Transformer Framework** (FANE + Swin Transformer)  
- **High interpretability** using visual attention maps (Grad-CAM support)  
- **Robust performance** under occlusions, illumination variations, and natural aging effects  
- **Lightweight deployment** suitable for cloud or edge-based elderly care systems  
- **Cross-platform compatibility** (PyTorch, CUDA, CPU/GPU)  

---

## 📂 **Repository Structure**

📦 Swin-FANE-Elderly-Emotion-Recognition
├── 📁 Models
│ ├── swin_transformer.pth
│ ├── swin_fane_best.pth
├── 📁 datasets
│ └── FANE
├── 📁 utilities
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── preprocess_data.py
├── 📁 images
│ ├── swin_fane_architecture.png
│ ├── attention_visualization.jpg
│ └── confusion_matrix.png
├── 📄 README.md
├── 📄 usage_guide.md
├── 📄 LICENSE
└── 📄 requirements.txt

yaml
Copy code

---

## 🧩 **Dataset**

### FANE Dataset (Facial Attention Network Embedding)
- A curated collection emphasizing **elderly facial emotion recognition**.  
- Contains high-resolution RGB facial images with diverse illumination, pose, and expression variations.  
- Covers seven primary emotions commonly observed in real-world interactions.  

Dataset preprocessing includes:
- Face detection (MTCNN / RetinaFace)  
- Image normalization to `224×224`  
- Data augmentation: rotation, flipping, brightness shift  

---

## ⚙️ **Installation Guide**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sanskarparab/Swin-FANE-Elderly-Emotion-Recognition.git
cd Swin-FANE-Elderly-Emotion-Recognition
2️⃣ Create Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows
3️⃣ Install Requirements
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
4️⃣ Run Training
bash
Copy code
python utilities/train_model.py --model swin_fane --epochs 25 --batch_size 32
5️⃣ Evaluate the Model
bash
Copy code
python utilities/evaluate_model.py
🧮 Results Summary
Model	Accuracy (%)	Precision (%)	F1-Score (%)
VGG16	87.2	86.4	85.9
ResNet50	89.5	88.6	88.1
Vision Transformer	91.1	90.4	90.0
Swin-FANE (Proposed)	93.4	92.8	93.0

📊 The Swin-FANE model achieved the highest accuracy, effectively handling age-related variations and subtle emotional cues.

📷 Visualizations
Architecture Overview

Attention Heatmaps

Confusion Matrix

🧠 Applications
Elderly Care Facilities – emotion monitoring for empathetic response systems

Healthcare Robotics – visual understanding for patient support robots

Affective Computing Research – benchmarking hybrid deep models for aged faces

Smart Homes – real-time well-being assessment through non-intrusive vision systems

🧾 Citation
If you use this repository or model in your research, please cite:

graphql
Copy code
@article{parab2025swinfane,
  title={Swin-FANE: Vision-Based Emotion Recognition Framework for Elderly Care},
  author={Sanskar Parab and Tanisha Saha},
  journal={Under Review, IEEE Transactions on Affective Computing},
  year={2025},
  institution={Somaiya Vidyavihar University, Mumbai}
}