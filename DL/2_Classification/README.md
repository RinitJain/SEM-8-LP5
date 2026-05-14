# Practical 2: Letter Recognition Classification using Deep Neural Network

## Objective

Build a Deep Neural Network (DNN) that classifies handwritten capital letters (A–Z) based on 16 numerical features extracted from printed characters. This is a **multi-class classification** problem — given features of a letter image, the model predicts which letter it is (one of 26 categories).

---

## Theory / Background

### What is Classification?
Classification is about predicting **which category** an input belongs to. Unlike regression (which predicts a number), classification outputs a class label. Here, we predict which letter (A–Z) a given set of features represents.

### What is Multi-class Classification?
When there are more than 2 possible output classes (like all 26 letters), it is called multi-class classification. The network outputs a **probability for each class**, and we pick the one with the highest probability.

### Key Terms You Must Know:
| Term | Plain English Meaning |
|---|---|
| **Softmax** | Activation function that converts raw outputs into probabilities that sum to 1. Used at the output layer for multi-class classification. |
| **Sparse Categorical Cross-Entropy** | Loss function for multi-class classification when labels are integers (0, 1, 2…). Measures how wrong the predicted probabilities are. |
| **LabelEncoder** | Converts text labels (A, B, C…) into numbers (0, 1, 2…) that the model can understand. |
| **Confusion Matrix** | A table showing how many times each class was correctly predicted and what it was confused with. |
| **Accuracy** | Percentage of correct predictions out of all predictions. |
| **Epoch** | One full pass through the training dataset. |
| **Dense Layer** | Fully connected layer — every neuron connects to every neuron in the next layer. |

### Why Softmax at Output?
If our output layer has 26 neurons (one per letter), softmax converts those 26 raw numbers into 26 probabilities that **sum to 1**. For example: [0.02, 0.01, ..., 0.85, ...]. The letter with the highest probability is the prediction.

### Why Sparse Categorical Cross-Entropy?
- **Categorical Cross-Entropy** is used when labels are one-hot encoded (e.g., [0,0,1,0,...])
- **Sparse Categorical Cross-Entropy** is used when labels are plain integers (e.g., 2 for 'C'). It is equivalent but avoids creating a 26-column one-hot array, saving memory.

---

## Dataset

**File:** `letter-recognition.data` (no header row — we add column names manually)

**Source:** UCI Machine Learning Repository — Letter Recognition Dataset

**Size:** 20,000 rows × 17 columns (16 features + 1 label)

**Target Column:** `letter` — the capital letter (A–Z) represented by each row

| Feature | Meaning |
|---|---|
| letter | Target: capital letter A–Z |
| x1–x16 | 16 numerical features extracted from scanned letter images (width, height, edge counts, correlation statistics, etc.) |

The 16 features capture the **visual statistics** of each printed character — things like how many horizontal/vertical edges it has, its width, height, and distribution of pixels. No actual image pixels are stored — only the statistical summary.

---

## Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
```
Standard imports for data handling, model building, and visualization.

### Step 2: Load Dataset
```python
df = pd.read_csv('letter-recognition.data', header=None)
df.columns = ['letter','x1','x2','x3','x4','x5','x6','x7','x8',
              'x9','x10','x11','x12','x13','x14','x15','x16']
```
The file has no header row (`header=None`), so we manually assign column names. The first column is the letter (target), the rest are the 16 features.

### Step 3: Separate Features and Labels
```python
X = df.drop('letter', axis=1)   # 16 input features (20000 × 16)
y = df['letter']                  # target labels (A–Z)
```

### Step 4: Encode Labels
```python
encoder = LabelEncoder()
y = encoder.fit_transform(y)
```
Letters A–Z are strings. The network needs numbers. LabelEncoder converts: A→0, B→1, C→2, ..., Z→25. Now `y` contains integers from 0 to 25.

### Step 5: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
80% training (16,000 rows), 20% testing (4,000 rows).

### Step 6: Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Scales all 16 features to mean=0, std=1. Fit only on training data, transform both training and test data.

### Step 7: Build the Neural Network
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(16,)))  # Hidden layer 1
model.add(Dense(32, activation='relu'))                      # Hidden layer 2
model.add(Dense(26, activation='softmax'))                   # Output: 26 classes
```
- **Input:** 16 features
- **Hidden Layer 1:** 64 neurons, ReLU — learns basic patterns
- **Hidden Layer 2:** 32 neurons, ReLU — learns higher-level combinations
- **Output Layer:** 26 neurons, Softmax — one probability per letter

Total parameters: **4,026 trainable weights**

### Step 8: Compile the Model
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- `sparse_categorical_crossentropy` because labels are integers (0–25), not one-hot
- Tracks accuracy as the performance metric

### Step 9: Train the Model
```python
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
```
- **10 epochs** of training
- `validation_split=0.1`: 10% of training data held out to track validation accuracy each epoch

### Step 10: Evaluate the Model
```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
```
Reports accuracy on the 4,000 unseen test samples.

### Step 11: Plot Training History
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
```
Plots how accuracy improved over 10 epochs for both training and validation data. If both curves rise together → good. If training rises but validation drops → overfitting.

### Step 12: Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)   # Convert probabilities to predicted class (0–25)
cm = confusion_matrix(y_test, y_pred)
```
`argmax` picks the index of the highest probability — that's the predicted letter. The confusion matrix is a 26×26 table showing which letters were confused with which.

---

## How to Run

1. Open `DL_02.ipynb` in Jupyter or Google Colab
2. Upload `letter-recognition.data` to the session directory
3. Run all cells top to bottom

---

## Output Explanation

| Metric | Value | What it Means |
|---|---|---|
| **Training Accuracy** | ~89.51% | Model correctly identified ~89.5% of letters during training |
| **Test Accuracy** | ~89.47% | On unseen data, ~89.5% of letters were correctly classified |
| **Test Loss** | ~0.33 | Low cross-entropy loss indicates good probability estimates |

**Training vs Validation Accuracy Graph:**
Both training and validation accuracy curves rise steadily over 10 epochs and converge close to each other, indicating the model is **not overfitting** — it generalizes well to new data.

**Confusion Matrix:**
A 26×26 matrix. The diagonal cells (top-left to bottom-right) show correct predictions. Off-diagonal cells show mistakes. For example, some letters like 'D' might be confused with 'O', or 'I' with 'l' because they look similar visually.

---

## Conclusion

We built a 3-layer DNN (64→32→26) to classify capital letters A–Z with ~89.5% accuracy using only 16 statistical features. The model learned to distinguish between 26 classes effectively. 

Key takeaways:
- Softmax output converts raw scores to probabilities — the highest probability is the predicted class
- LabelEncoder transforms string labels to integers efficiently
- 89.5% accuracy on 26 classes (random guessing = 3.8%) shows the model learned meaningful patterns
- Training and validation accuracy being close together shows no overfitting

---

## Viva Questions & Answers

**Q1: What is the difference between binary classification and multi-class classification?**
> Binary classification has only 2 output classes (e.g., spam/not spam, yes/no). Multi-class classification has more than 2 classes. Here, we have 26 classes (A–Z). Binary uses sigmoid at output; multi-class uses softmax.

**Q2: What is softmax and why is it used at the output layer?**
> Softmax converts the 26 raw output numbers into 26 probabilities that sum to 1. For example, if the model is 85% sure the letter is 'C', softmax gives P(C) = 0.85. We pick the class with the highest probability. Without softmax, the raw numbers wouldn't represent probabilities.

**Q3: What is LabelEncoder? Why not use One-Hot Encoding?**
> LabelEncoder converts categories (A–Z) to integers (0–25). One-hot encoding creates a binary vector of size 26 for each sample. LabelEncoder is more memory-efficient and works directly with `sparse_categorical_crossentropy`. One-hot would work with `categorical_crossentropy`.

**Q4: What is sparse_categorical_crossentropy?**
> It is the loss function for multi-class classification when labels are integers. It measures the difference between the predicted probability distribution and the true class. The "sparse" part means it expects integer labels (like 5), not one-hot vectors (like [0,0,0,0,0,1,...]).

**Q5: What is a confusion matrix?**
> A 26×26 table where rows represent actual classes and columns represent predicted classes. The diagonal shows correct predictions. Off-diagonal entries show what the model confused each letter with. A perfect model would have all values on the diagonal.

**Q6: What does accuracy mean and how is it calculated?**
> Accuracy = (Number of correct predictions) / (Total predictions) × 100. Our model got 89.47% accuracy meaning out of 4,000 test samples, about 3,579 were correctly classified.

**Q7: Why are training and validation accuracy both important?**
> Training accuracy shows how well the model learned the training data. Validation accuracy shows if it can generalize. If training accuracy is 95% but validation is 70%, the model has overfit — it memorized training data instead of learning patterns.

**Q8: What is the vanishing gradient problem?**
> In deep networks with sigmoid activation, gradients (error signals used to update weights) become very small as they travel back through layers. This makes early layers learn very slowly. ReLU solves this because its gradient is always 1 for positive inputs, so the signal doesn't shrink.

**Q9: What are the 16 features in this dataset?**
> They are statistical features extracted from scanned images of printed capital letters: things like the width and height of the letter, number of horizontal/vertical edges, mean position of pixels (x, y coordinates), variance, etc. No raw pixel data is stored.

**Q10: What happens if you increase epochs too much?**
> The model might overfit — it learns the training data too perfectly, including noise, and performs worse on new data. The validation accuracy will start decreasing even as training accuracy keeps rising.

**Q11: What is batch size and why does it matter?**
> Batch size is how many samples are processed before the model's weights are updated. Smaller batch (like 16) → more frequent but noisier updates. Larger batch (like 256) → smoother updates but slower learning. Batch size 32 is a common balanced choice.

**Q12: What is the role of the hidden layers?**
> The first hidden layer (64 neurons) learns basic patterns (combinations of features). The second hidden layer (32 neurons) learns more abstract, higher-level patterns from the first layer's output. Together they allow the model to learn complex non-linear relationships between features and letter classes.
