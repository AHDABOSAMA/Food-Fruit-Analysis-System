# 🍎 Food & Fruit Analysis System

Deep Learning-based Computer Vision project for food and fruit understanding using classification, similarity learning, and semantic segmentation.

---

## 👨‍💻 Team Members

* Abdelrahman Waheed 
* John Polis 
* Losia Awny 
* Ahdab Osama 
* Youssef Ashraf
* Fady Hany 

---

## 🚀 Project Overview

This project is a multi-task Computer Vision system that performs:

* 🍎 Binary classification (Food vs Fruit)
* 🍓 Multi-class fruit classification (30 classes)
* 🧠 Few-shot learning using Siamese Networks
* 🖼️ Binary semantic segmentation (UNet & DeepLabV3+)
* 🎯 Multi-class semantic segmentation

Built using **PyTorch, OpenCV, and pretrained CNN architectures**.

---

## 🧠 Models Used

### 1. Classification Models

* ResNet18 (Transfer Learning)
* Fully Connected layer modified for task-specific outputs

### 2. Siamese Network

* ResNet18 backbone
* Triplet Loss
* 128D embedding space

### 3. Segmentation Models

* UNet (with ResNet34 encoder)
* DeepLabV3+ (with ResNet50 encoder)

---

## 📊 Project Parts

### 🔹 Part A: Food vs Fruit Classification

* Model: ResNet18
* Accuracy: 100%
* Transfer learning from ImageNet

### 🔹 Part B: Siamese Network (Few-Shot Learning)

* Triplet loss-based training
* Embedding similarity learning
* Validation Accuracy: ~72.8%

### 🔹 Part C: 30-Class Fruit Classification

* Data augmentation applied
* ResNet18 fine-tuning
* Accuracy: 100%

### 🔹 Part D: Binary Segmentation

* UNet vs DeepLabV3+
* Pixel-level fruit detection
* Best validation loss: 0.0273 (UNet)

### 🔹 Part E: Multi-Class Segmentation

* UNet (ResNet34)
* DeepLabV3+ (ResNet50)
* Best accuracy: 97.92%

---

## 🛠️ Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Torchvision
* Transfer Learning

---

## 📂 Project Structure

```
Food-Fruit-Analysis-System/
│
├── Part_A/
├── Part_B/
├── Part_C/
├── Part_D/
├── Part_E/
├── utils/
├── requirements.txt
├── README.md
```

---

## 📦 Dataset

Due to size limitations (2.1GB), datasets are not included in this repository.




## 📈 Results Summary

| Task                  | Performance |
| --------------------- | ----------- |
| Binary Classification | 100%        |
| Fruit Classification  | 100%        |
| Siamese Network       | 72.8%       |
| Segmentation (UNet)   | 97.9%       |

---

## 💡 Key Highlights

* Transfer learning with ResNet architectures
* Few-shot learning using Siamese Networks
* Pixel-level segmentation using UNet & DeepLabV3+
* Multi-task computer vision pipeline

---

## 📌 Notes

* Large dataset and model files are excluded due to GitHub size limits
* External storage (Google Drive) is used for datasets and trained models

---

## 📜 License

This project is for academic and educational purposes.

---

## ⭐ Acknowledgements

Special thanks to the instructors and datasets used in this project.
