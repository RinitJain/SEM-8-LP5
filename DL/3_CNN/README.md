# Practical 3: Fashion MNIST Image Classification using Convolutional Neural Network (CNN)

## Objective

Build a Convolutional Neural Network (CNN) that looks at 28×28 grayscale images of clothing items and classifies them into one of 10 categories (T-shirt, Trouser, Pullover, etc.). This is an **image classification** problem — the model must learn to recognize visual patterns from pixel values.

---

## Theory / Background

### What is a CNN?
A Convolutional Neural Network (CNN) is a type of deep learning model designed specifically for images. Unlike a regular Dense (fully connected) network that treats every pixel independently, CNNs use **convolution filters** that slide across the image and detect local patterns like edges, corners, and shapes.

### Why Not Just Use a Dense Network for Images?
A 28×28 image has 784 pixels. If we flatten it and feed it into a Dense layer with 64 neurons, that's 784 × 64 = ~50,000 weights just for one layer. For larger images this explodes. Also, a Dense layer doesn't understand spatial relationships — it doesn't know that two adjacent pixels are neighbors.

CNNs solve this by:
1. **Sharing weights** (a single filter applies to all parts of the image)
2. **Preserving spatial structure** (nearby pixels are processed together)

### Key Terms You Must Know:
| Term | Plain English Meaning |
|---|---|
| **Convolution (Conv2D)** | Slide a small filter (e.g., 3×3) across the image and compute a dot product at each position. Detects features like edges, curves |
| **Filter / Kernel** | A small matrix of weights (e.g., 3×3) that slides across the image to detect a specific pattern |
| **Feature Map** | The output of applying one filter across the entire image — shows where that feature (edge, curve) was found |
| **MaxPooling** | Takes the maximum value from a small region (e.g., 2×2). Reduces image size while keeping the most important features |
| **Flatten** | Converts the 2D feature maps into a 1D vector so it can be fed into Dense layers |
| **Dropout** | Randomly sets some neuron outputs to 0 during training to prevent overfitting |
| **Softmax** | Output activation for multi-class classification — converts scores to probabilities |
| **Normalization (÷255)** | Pixel values range 0–255; dividing by 255 scales them to 0–1, which makes training stable |

### How Convolution Works (Simple Explanation):
Imagine a 3×3 filter looking for vertical edges. It slides across the 28×28 image, one position at a time. At each position, it multiplies its 9 values with the 9 pixels underneath and sums them up. High sum = that feature is present here. This creates a new "feature map" that highlights where vertical edges exist.

### How MaxPooling Works:
After convolution, we have a large feature map. MaxPooling with 2×2 divides it into 2×2 blocks and keeps only the maximum value from each block. This:
- Reduces the size (28×28 → 14×14 after pooling)
- Makes the model less sensitive to exact pixel positions (translation invariance)

### What is Dropout?
Dropout randomly deactivates 20% of neurons during each training step. This forces the remaining neurons to work harder and not depend on specific neurons. The result is a more robust, generalized model. Dropout is only active during training — during prediction/testing, all neurons are used.

---

## Dataset

**Files:** `fashion-mnist_train.csv` (60,000 rows) and `fashion-mnist_test.csv` (10,000 rows)

**Source:** Zalando Research — Fashion-MNIST dataset

**Size:** Each image is 28×28 = 784 pixels (grayscale, 0–255 values)

**Class Labels (10 categories):**
| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Each row in the CSV = one image. First column = label (0–9). Next 784 columns = pixel values of the 28×28 image stored as a flat row.

---

## Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
```
- `Conv2D`, `MaxPooling2D` — CNN-specific layers
- `Flatten` — converts 2D maps to 1D
- `Dropout` — regularization to prevent overfitting
- `confusion_matrix`, `classification_report` — evaluation tools

### Step 2: Load Data from CSV
```python
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')
```
Reads the fashion-mnist CSV files into pandas DataFrames.

### Step 3: Separate Features and Labels
```python
X_train = train_df.iloc[:, 1:].values   # All columns except first (pixel values)
y_train = train_df.iloc[:, 0].values    # First column = labels

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values
```
`iloc[:, 1:]` = all rows, all columns from index 1 onward (skipping the label column).
`iloc[:, 0]` = all rows, only the first column (the label).

### Step 4: Normalize Pixel Values
```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```
Pixels are 0–255 integers. Dividing by 255 scales them to 0.0–1.0. This makes gradient computations stable and training faster. Neural networks train much better with small input values.

### Step 5: Reshape for CNN
```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```
The CNN expects 4D input: (batch_size, height, width, channels).
- `-1` = infer batch size automatically (60,000 for training)
- `28, 28` = image height and width
- `1` = 1 color channel (grayscale). RGB images would have 3 channels.

### Step 6: Build the CNN Model
```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
```
**Conv2D layer:** 32 filters, each of size 3×3. Each filter slides across the 28×28 image and creates a feature map. With 32 filters, we get 32 feature maps. `activation='relu'` is applied element-wise to the feature maps.

Output after Conv2D: 26×26×32 (image shrinks by 2 because 3×3 filter on 28×28 gives 26×26)

```python
model.add(MaxPooling2D(2,2))
```
**MaxPooling2D:** 2×2 pool, reduces each feature map from 26×26 → 13×13.
Output after pooling: 13×13×32

```python
model.add(Flatten())
```
**Flatten:** Converts 13×13×32 = 5,408 values into a 1D vector of 5,408 values. Now it can be fed into Dense layers.

```python
model.add(Dense(64, activation='relu'))
```
**Dense hidden layer:** 64 neurons, learns high-level patterns from the flattened feature maps.

```python
model.add(Dropout(0.2))
```
**Dropout 20%:** During each training step, randomly sets 20% of the 64 neurons' outputs to 0. Prevents over-reliance on any specific neuron.

```python
model.add(Dense(10, activation='softmax'))
```
**Output layer:** 10 neurons (one per clothing category), softmax gives probabilities for each class.

**Total Parameters:** 347,146 trainable weights

### Step 7: Compile the Model
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
Same as classification: `sparse_categorical_crossentropy` because labels are integers (0–9).

### Step 8: Train the Model
```python
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
```
Only **5 epochs** — CNNs learn efficiently from images. With more data (60,000 images) even 5 epochs give good results. 90% of training data used for training, 10% for validation.

### Step 9: Evaluate the Model
```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```
Tests on 10,000 unseen images.

### Step 10: Visualize Training
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```
Shows accuracy curve over 5 epochs for both training and validation.

### Step 11: Sample Predictions
```python
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
y_pred = np.argmax(model.predict(X_test), axis=1)
```
`model.predict()` returns probabilities for all 10 classes. `argmax` picks the class with the highest probability.

Displays 5 test images with their actual (T:) and predicted (P:) labels side by side.

### Step 12: Classification Report & Confusion Matrix
```python
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
```
Classification report shows precision, recall, and F1-score for each of the 10 classes. Confusion matrix shows a 10×10 grid of correct vs incorrect predictions per class.

---

## How to Run

1. Open `DL_03.ipynb` in Jupyter or Google Colab
2. Upload `fashion-mnist_train.csv` and `fashion-mnist_test.csv` to the session
3. Run all cells top to bottom

---

## Output Explanation

| Metric | Value | What it Means |
|---|---|---|
| **Training Accuracy** | ~90.09% | Model correctly classifies ~90% of training images after 5 epochs |
| **Test Accuracy** | ~86.96% | On 10,000 unseen images, ~87% are correctly classified |
| **Test Loss** | ~0.38 | Cross-entropy loss on test data |

**Training vs Validation Accuracy Plot:**
Both curves rise over 5 epochs. Training accuracy is slightly higher than validation, which is normal. No severe overfitting.

**Sample Prediction Output:**
```
T: T-shirt   P: T-shirt    ✓
T: Trouser   P: Trouser    ✓
T: Pullover  P: Coat       ✗  (common confusion between similar shapes)
```

**Classification Report — most confusable classes:**
- Shirt vs T-shirt: both have similar shapes
- Pullover vs Coat: similar silhouettes
- Sneaker vs Ankle boot: similar shoe shapes

**Confusion Matrix:**
The 10×10 matrix shows most predictions fall on the diagonal (correct). Shirt (class 6) has the most confusion with other classes.

---

## Conclusion

We built a CNN with one convolutional layer, one pooling layer, and two dense layers. It achieved ~87% accuracy on 10,000 test images of clothing items from 10 categories.

Key takeaways:
- CNNs are far better suited to image data than plain Dense networks because they detect local patterns via convolution
- MaxPooling reduces computation while preserving important features
- Dropout prevented overfitting — without it, the model would score higher on training but lower on test
- Normalizing pixels (÷255) was essential for stable training
- Even with just 5 epochs, CNNs achieve good accuracy on images

---

## Viva Questions & Answers

**Q1: What is a Convolutional Neural Network (CNN)?**
> A CNN is a deep learning model designed for images. It uses convolutional layers that apply small filters across the image to detect patterns like edges, curves, and shapes. Unlike Dense networks, CNNs share weights across positions, making them efficient and spatially aware.

**Q2: What is a convolution operation?**
> A convolution applies a small filter (e.g., 3×3 matrix of weights) across an image by sliding it one position at a time. At each position, it multiplies the filter values by the overlapping pixel values and sums them. This produces one value in the output feature map. It detects whether a particular pattern (like an edge) is present at that location.

**Q3: What is a feature map?**
> A feature map is the output produced when one filter is applied across the entire image. It shows where in the image that particular feature was detected. With 32 filters, we get 32 feature maps — each highlighting different visual patterns.

**Q4: What is MaxPooling and why is it used?**
> MaxPooling takes a small region (2×2) and keeps only the maximum value. It reduces the spatial dimensions of feature maps (e.g., 26×26 → 13×13), reducing computation and making the model less sensitive to exact pixel positions. This is called translation invariance — the model recognizes a feature regardless of where exactly in the image it appears.

**Q5: Why do we reshape the input to (28, 28, 1)?**
> CNNs require 4D input: (samples, height, width, channels). Height=28, Width=28 is the image size. Channels=1 because Fashion-MNIST images are grayscale (one intensity channel). RGB images would have 3 channels.

**Q6: What is Dropout and why is it used?**
> Dropout randomly deactivates a fraction (here, 20%) of neurons during each training step. This prevents neurons from co-adapting — where one neuron depends on specific others. The result is a more general model that performs better on unseen data. Dropout is disabled during prediction/testing.

**Q7: Why divide pixel values by 255?**
> Pixel values are integers from 0 to 255. Neural networks train better with small values (0–1 range). Large input values cause large gradients, making training unstable. Dividing by 255 normalizes all pixels to the 0.0–1.0 range.

**Q8: What does Flatten do?**
> After convolutional and pooling layers, data is in a 2D/3D shape (e.g., 13×13×32). Dense layers need 1D input. Flatten converts this 3D volume into a single 1D vector (13×13×32 = 5,408 values) that can be fed into the Dense layers.

**Q9: Why is CNN better than a plain Dense network for images?**
> Dense networks treat all 784 pixels as independent inputs — they don't know which pixels are neighbors. CNNs apply filters that look at local regions of pixels together, learning spatial patterns. CNN weights are shared across all positions, making them much more parameter-efficient and effective for image data.

**Q10: What are precision, recall, and F1-score?**
> - **Precision:** Of all predictions for class X, how many were actually class X? (Avoid false positives)
> - **Recall:** Of all actual class X items, how many did we correctly identify? (Avoid false negatives)
> - **F1-Score:** Harmonic mean of precision and recall — a single balanced metric. F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Q11: What is the architecture of the CNN used here?**
> Input (28×28×1) → Conv2D(32 filters, 3×3, ReLU) → MaxPool(2×2) → Flatten → Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)

**Q12: What are the 10 Fashion-MNIST classes?**
> T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Sandal (5), Shirt (6), Sneaker (7), Bag (8), Ankle boot (9). The dataset was created by Zalando as a more challenging replacement for the original handwritten digit MNIST dataset.
