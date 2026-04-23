# Machine Learning Basics Notes

## 1. What is Image Classification?
Image classification is the task of predicting the category (label) of an image.

Example:
- Input: Image of a show
- Output: "Shoe"

## 2. Dataset
A dataset is a collection of labeled images.

Each data point:
- Image
- Label (correct answer)

---

## 3. Train / Validation / Test

- Training Set: model learns from this
- Validation Set: used to tune model
- Test Set: final evaluation (never seen before)

---

## 4. Epoch
One full pass through the training dataset.

---

## 5. Batch
A small subset of data processed at once.

Example:
- Dataset = 60,000 images
- Bath size = 100
- 600 batches per epoch

---

## 6. Loss
Measures how wrong the model is.

Goal: minimize loss

---

## 7. Optimizer
Updates the model to reduce loss.

Example:
- SGD
- Adam

---

## 8. Overfitting
Model memorizes training data but performs poorly on new data.

Signs:
- Training accuracy high
- Validation accuracy low

---

## 9. Underfitting
Model is too simple and performs poorly everywhere.

---

## 10. Neural Network
A function that learns patterns from data using layers.

---

## 11. MLP (Multiplayer Perceptron)
A basic neural network using fully connected layers.

Weak for images because:
- It ignores spatial structure

---

## 12. CNN (Convolutional Neural Network)
Special type of neural network for images

Key idea:
- Looks at small regions of image
- Detects patterns (edges, shapes, textures)

---

## 13. Why CNN > MLP for images?
CNN keeps spatial information:
- MLP flattens image: loses structure
- CNN keeps image layout: better performance