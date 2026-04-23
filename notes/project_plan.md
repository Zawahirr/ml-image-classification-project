# Project Plan

## 1. Goal
The goal of this project is to compare different neural network architectures for image classification and analyze their performance.

---

## 2. Datasets

### Fashion-MNIST
- 28x28 grayscale images
- 10 classes (clothing items)
- Easier dataset

### CIFAR-10
- 32x32 color images
- 10 classes (objects like cars, dogs, etc.)
- More complex dataset

---

## 3. Models

### 1. MLP (Baseline)
- Fully connected neural network
- Flattens image input
- Used as a comparison baseline

### 2. Simple CNN
- Basic convolutional neural network
- Learns spatial features

### 3. LeNet
- Classic CNN architecture
- Well-known benchmark model

### 4. Improved CNN (Custom)
- Based on Simple CNN
- Will include:
  - More convolution layers
  - Dropout
  - Possibly Batch Normalization

---

## 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 5. Experiments

For each dataset:
- Train all models
- Compare performance
- Analyze results

---

## 6. Expected Outcomes
- CNNs should outperform MLP
- Improved CNN should perform better than Simple CNN
- CIFAR-10 will be harder than Fashion-MNIST

---

## 7. Learning Goals
- Understand CNNs deeply
- Learn model training pipeline
- Learn how to evaluate models properly
- Learn how to present ML results professionally