# Visual Defect Detection in Automotive Components using AI Models

## Abstract

This project aims to automate the visual inspection of automotive components using deep learning. It explores lightweight CNN models (MobileNetV2), Vision Transformers (ViT), and a hybrid CNN–Transformer architecture to identify manufacturing defects. The system is designed to support real-time inspection, improve consistency over manual inspection, and provide model explainability using Grad-CAM.

## Problem Statement

Manual defect inspection in automotive manufacturing is time-consuming, inconsistent, and prone to human error. Traditional machine vision methods struggle with variations in lighting, surface texture, and defect patterns.
This project seeks to build a scalable and accurate AI-based defect detection system capable of classifying multiple defect types while providing interpretable visual explanations for each prediction.

# 📂 Project File Structure

For quick reference, here is an overview of the main directories in this project:

| Directory | Description |
| :--- | :--- |
| `data/` | **Dataset** files (raw samples, prepared splits, etc.) |
| `notebooks/` | **Jupyter/Colab workflows** for experimentation and exploration. |
| `src/` | **Main source code** (modules for models, training, preprocessing, etc.). |
| `models/` | Trained **model weights** and checkpoints. *(Typically ignored by Git)* |
| `results/` | **Logs, metrics, and outputs** from training and evaluation runs. |
| `figures/` | **Plots and visualizations** generated during analysis. |
| `docs/` | Project **documentation** (e.g., usage guides, API reference). |
| `scripts/` | **Utility scripts** for setup, data tasks, or automation. |

---

## Dataset Access

The full dataset used for training is stored on Google Drive (not included in the GitHub repository due to size).

🔗 **Dataset Link:**
[Click here to access the dataset](https://drive.google.com/drive/folders/15YNVf6SXEYlRN6jbVMrr2GVh-7qZEaNn?usp=sharing)

**This dataset contains:**
2 defect categories + 1 accept category
Preprocessed and structured folders (train, val, test)
~17k images

###To use the dataset in Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/MyDrive/Classification_Data_FINAL"
