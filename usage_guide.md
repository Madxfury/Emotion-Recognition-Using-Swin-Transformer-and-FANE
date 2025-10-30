# 🧭 Usage Guide for Swin-FANE: Vision-Based Emotion Recognition for Elderly Care

This guide provides a clear, step-by-step explanation on how to set up, train, and evaluate the **Swin-FANE** model — a hybrid attention-transformer framework designed for elderly emotion recognition.

---

## 📦 Dataset Preparation

### FANE Dataset (Facial Attention Network Embedding)

The **FANE dataset** is used in this project. It contains facial images of elderly individuals under various emotional states. Each image is annotated with a ground truth emotion label.  

Ensure the dataset directory follows this structure:

📁 datasets/FANE
├── Angry
├── Disgust
├── Fear
├── Happy
├── Neutral
├── Sad
└── Surprise

bash
Copy code

### Preprocessing the Dataset

Use the provided preprocessing script to split your dataset into training, validation, and testing sets:

```bash
python utilities/preprocess_data.py
This will automatically:

Convert all images to RGB

Resize to 224×224

Normalize using ImageNet statistics

Create a split ratio of 80% (train), 10% (validation), and 10% (test)

🚀 Training the Model
Step 1: Select the Model
The repository supports multiple model architectures such as:

Swin-FANE (Proposed)

Vision Transformer (ViT)

ResNet50

VGG16

Select the model type in the training command.

Step 2: Train the Model
bash
Copy code
python utilities/train_model.py --model swin_fane --epochs 25 --batch_size 32
Optional Parameters:

Parameter	Description	Default
--model	Choose architecture (swin_fane, vit, resnet50, etc.)	swin_fane
--epochs	Number of epochs for training	25
--batch_size	Training batch size	32
--learning_rate	Learning rate for optimizer	1e-4

Trained model weights will be saved in the Models/ directory as:

Copy code
Models/Swin_FANE_Best_Model.pth
📊 Model Evaluation
Step 1: Evaluate Trained Model
Once the model is trained, evaluate it on the test dataset:

bash
Copy code
python utilities/evaluate_model.py
Step 2: Output Metrics
The evaluation script provides the following metrics:

Overall Accuracy

Precision

Recall

F1-Score (Macro and Weighted)

Classification Report per Emotion

Confusion Matrix (optional visualization)

Example output:

sql
Copy code
📊 FINAL TEST SET ACCURACY: 93.4%
Precision (avg): 92.8%
F1-Score (avg): 93.0%
🔍 Visualization and Analysis
Grad-CAM Visualization
To visualize which regions influenced the model’s decision:

bash
Copy code
python utilities/visualize_results.py --model swin_fane --checkpoint Models/Swin_FANE_Best_Model.pth
This will generate Grad-CAM heatmaps for each emotion class, showing key facial regions such as eyes, mouth, and eyebrows.

Confusion Matrix
Confusion matrices can be plotted directly in Jupyter or saved as an image file during evaluation for report documentation.

⚙️ Fine-Tuning and Optimization
Adjusting Hyperparameters
Modify the hyperparameters in train_model.py:

python
Copy code
learning_rate = 1e-4
dropout_rate = 0.6
weight_decay = 1e-5
Layer Freezing Strategy (Recommended for Transformers)
Freeze the initial Swin Transformer layers for stable training during early epochs:

python
Copy code
for name, param in model.backbone.named_parameters():
    if 'layers.0' in name or 'layers.1' in name:
        param.requires_grad = False
Unfreeze after 10 epochs to fine-tune deeper representations:

python
Copy code
if epoch == 10:
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True
💡 Tips for Best Results
Use Balanced Splits – Maintain proportional emotion distribution across train/val/test.

Augment Data – Apply random rotations, flips, and brightness changes to improve generalization.

Monitor Overfitting – Enable early stopping if validation loss stagnates for more than 5–10 epochs.

Visualize Attention – Inspect Grad-CAM results to validate model focus areas.

GPU Recommended – Training on CUDA-enabled GPU significantly speeds up training.

🌐 References
FANE Dataset – Facial Attention Network Embedding for Elderly Emotion Analysis

Swin Transformer – Liu et al., Microsoft Research, 2021

Vision Transformer (ViT) – Dosovitskiy et al., Google Research, 2020

PyTorch Documentation – https://pytorch.org/docs/stable/index.html

For additional support, refer to the main README.md or contact:
📩 Sanskar Parab – sanskarparab@somaiya.edu
📩 Tanisha Saha – tanishasaha@somaiya.edu

yaml
Copy code

---

Would you like me to also generate a **`utilities/preprocess_data.py` example script** to match this guide (train/val/test split + augmentation)?  
That would make your repo fully reproducible and GitHub-review ready.