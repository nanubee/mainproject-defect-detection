# 🔍 Intelligent Visual Defect Detection in Automotive Brake Discs Using Lightweight AI Models

An end-to-end computer vision and deep learning system designed for automated quality control in industrial manufacturing. This project benchmarks lightweight Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and attention-augmented Hybrid architectures for detecting surface defects and anomalies in automotive brake disc components.

---

## 🌟 Key Features

* **Multi-Architecture Benchmarking:** Evaluates four deep learning paradigms:
  * **Baseline CNN:** MobileNetV2
  * **Baseline ViT:** Vision Transformer (`vit_small_patch16_224`)
  * **Standard Hybrid:** Feature fusion of MobileNetV2 + ViT
  * **Proposed ECA-Hybrid:** Attention-augmented hybrid model utilizing Efficient Channel Attention (ECA) modules.
* **Explainable AI (XAI):** Built-in universal Grad-CAM engine that dynamically handles spatial 2D feature maps from CNNs as well as 1D patch sequences from Vision Transformers to produce decision heatmap visualizations.
* **Interactive Streamlit Dashboard:** 
  * Real-time automated inspection simulation via the **Live Inspection** mode.
  * Manual file upload for on-demand component evaluation.
  * Adjustable safety threshold sensitivity slider.
  * Automated defect classification into **Accept**, **Casting Fault**, and **Surface Imperfection**.

---

## 📊 Evaluated Architectures

| Architecture | Model Class | Key Characteristics | Target Application |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | Baseline CNN | Lightweight depthwise separable convolutions | Ultra-fast baseline |
| **Baseline ViT** | Pure Transformer | Global self-attention over patch sequences (14x14 grid) | Global spatial context |
| **Standard Hybrid** | CNN + ViT | Concatenated 1280-D (CNN) and 384-D (ViT) feature vectors | Balanced local + global context |
| **ECA-Hybrid (Proposed)** | Attention-Hybrid | Dual ECA modules for cross-channel attention without dimensionality reduction | High precision quality control |

### Application Modes
* **🏭 LIVE INSPECTION (Random):** Randomly samples component images from the `Safe_Demo_Images/` directory to simulate automated conveyor belt camera feeds.
* **📤 Manual Upload:** Upload any `.jpg`, `.jpeg`, or `.png` brake disc image to evaluate custom inputs in real time.

## 🛠️ Tech Stack & AI Methodologies

* **Core Deep Learning Framework:** PyTorch
* **Computer Vision Ecosystem:** 
  * `torchvision` (For baseline CNNs and data augmentation pipelines)
  * `timm` / PyTorch Image Models (For state-of-the-art Vision Transformer implementation)
* **Neural Architectures & Mechanisms:**
  * **CNNs:** Depthwise separable convolutions (MobileNetV2)
  * **Transformers:** Multi-Head Self-Attention and 1D sequence patching (ViT-Small)
  * **Attention Modules:** Efficient Channel Attention (ECA) utilizing 1D convolutions for cross-channel interaction without dimensionality reduction.
  * **Feature Fusion:** Multi-modal concatenation of global and local feature vectors.
* **Explainable AI (XAI):** Custom Gradient-weighted Class Activation Mapping (Grad-CAM) engine, featuring advanced mathematical 1D-to-2D reshaping to extract spatial heatmaps from Transformer sequence patches.
* **Tensor Operations & Image Processing:** NumPy, OpenCV, Pillow (PIL)
* **Model Serving & Inference UI:** Streamlit (For rapid AI deployment and real-time inference testing)
