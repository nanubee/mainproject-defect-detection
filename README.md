Visual Defect Detection in Automotive Components using AI Models
Overview

This project focuses on building an AI-based system for detecting visual defects in automotive components. The goal is to replace manual inspection with a more accurate, consistent, and scalable deep learning solution.

The project uses CNNs (MobileNetV2), Transformers (ViT-Small), and a planned Hybrid Model to learn both local and global features of defect patterns.
Explainability will be achieved through Grad-CAM, and the final model will be integrated into a Streamlit dashboard for real-time inference.

Objectives

Develop a robust AI model for defect classification.

Explore MobileNet, ViT, and Hybrid CNN-Transformer architectures.

Build an explainability module using Grad-CAM.

Prepare a clean and modular codebase for development + teamwork.

Eventually create a Streamlit dashboard for real-time inspection.

📁 Project Folder Structure

Your current repository contains the following folders:

data/           # Dataset folder (images, samples, splits)
    samples/    # Small sample images for testing workflows

notebooks/      # Jupyter/Colab notebooks for experiments

src/            # Core source code
    data/       # Dataloaders, preprocessing, augmentations
    models/     # Model definitions (MobileNet, ViT, Hybrid)
    train.py    # Training script (to be created)
    evaluate.py # Evaluation script (to be created)
    inference.py# Inference script (optional)

models/         # Trained model weights (ignored by .gitignore)

results/        # Training logs, metrics, plots

figures/        # Images, diagrams, visualizations

docs/           # Project documentation, architecture details

scripts/        # Utility scripts (data prep, training, automation)


This structure is standard for collaborative deep learning projects and keeps the code organized and maintainable.

🧠 Models Planned

MobileNetV2: Lightweight CNN baseline for efficient feature extraction

ViT-Small: Vision Transformer for global attention-based understanding

Hybrid Model: CNN + Transformer fusion for improved performance

Grad-CAM: For model explainability

(Implementations will be added step-by-step.)

🛠 Requirements

A requirements.txt file will be added later with libraries such as:

PyTorch / TensorFlow

torchvision / timm

numpy, pandas

matplotlib

streamlit

albumentations

▶️ Usage (Work in Progress)

As the project is in setup phase:

Training scripts will go into src/train.py.

Dataloader will be in src/data/.

Models will be implemented in src/models/.

Experiments will be run inside /notebooks.

Instructions will be updated once the training pipeline is added.

👩‍💻 Contribution Guidelines

Create a new branch for each feature:

git checkout -b feat/your-feature-name


Commit changes frequently.

Push your branch and create a Pull Request for review.

📚 Acknowledgments

Guided by: Ms. Liya Joseph

Academic Project at: Rajagiri School of Engineering & Technology

Dataset Source: Roboflow (Casting/Brake Disc defect datasets)

Open-source contributors from PyTorch, TensorFlow, Streamlit
