# Practical 1: Boston Housing Price Prediction using Deep Neural Network (Regression)

## Objective

Build a Deep Neural Network (DNN) that takes 13 features of a house and predicts its price (in $1000s). This is a **regression** problem — the output is a continuous number, not a category. We use the famous Boston Housing dataset and compare how well the model predicts prices on unseen data.

---

## Theory / Background

### What is a Neural Network?
A Neural Network is a system inspired by the human brain. It consists of layers of "neurons" (mathematical units). Each neuron takes inputs, multiplies them by weights, adds a bias, and passes the result through an activation function. By training on data, the network automatically learns the right weights.

### What is Deep Learning?
When a neural network has more than one hidden layer, it is called a Deep Neural Network (DNN). The "depth" helps it learn more complex patterns in data.

### Key Terms You Must Know:
| Term | Plain English Meaning |
|---|---|
| **Layer (Dense)** | A fully connected layer where every neuron connects to every neuron in the next layer |
| **Activation (ReLU)** | ReLU = max(0, x). It introduces non-linearity. Negative values become 0, positives stay as-is. |
| **Loss (MSE)** | Mean Squared Error — average of squared differences between predicted and actual values. We minimize this during training. |
| **Optimizer (Adam)** | Adam automatically adjusts the learning rate to minimize loss efficiently. It is the most commonly used optimizer. |
| **Epoch** | One full pass through the entire training dataset |
| **Batch Size** | Number of samples processed before updating weights. Batch of 32 means weights update after every 32 samples. |
| **StandardScaler** | Converts each feature to have mean=0 and std=1. This prevents large-valued features from dominating the learning. |
| **MAE (Mean Absolute Error)** | Average absolute difference between actual and predicted values. More human-readable than MSE. |

### Why No Activation at the Output Layer?
For regression, we want the model to output any real number (like 23.5 or 47.2). If we used ReLU, all negatives would be cut off. If we used sigmoid, output would be forced between 0 and 1. So we use **no activation** (linear output) for regression tasks.

---

## Dataset

**File:** `HousingData.csv`

**Size:** 506 rows × 14 columns (13 features + 1 target)

**Target Column:** `MEDV` — Median value of owner-occupied homes in $1000s

| Feature | Meaning |
|---|---|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land for large lots |
| INDUS | Proportion of non-retail business acres |
| CHAS | Is the tract next to Charles River? (1=yes, 0=no) |
| NOX | Nitric oxide concentration (pollution) |
| RM | Average number of rooms per house |
| AGE | Proportion of old units built before 1940 |
| DIS | Distance to employment centers |
| RAD | Index of highway accessibility |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio (school quality) |
| B | Statistic involving proportion of Black residents |
| LSTAT | % of lower-status population |
| **MEDV** | **Target: Median house price ($1000s)** |

The dataset has **missing values** that are filled with the **column mean** before training.

---

## Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import pandas as pd
```
- `numpy` for math operations
- `matplotlib` for plotting graphs
- `tensorflow/keras` to build and train the neural network
- `sklearn` for data splitting and scaling
- `pandas` to read and manipulate the CSV file

### Step 2: Load the Dataset
```python
df = pd.read_csv('/content/HousingData.csv')
```
Reads the CSV file into a table called `df`. Has 506 rows and 14 columns.

### Step 3: Handle Missing Values
```python
df = df.fillna(df.mean())
```
Some cells in the CSV are empty (NaN). We fill them with the **average** of that column. This is called **mean imputation** — a simple way to handle missing data without removing rows.

### Step 4: Separate Features and Target
```python
X = df.drop('MEDV', axis=1)   # 13 input features
y = df['MEDV']                 # target house prices
```
`X` contains the 13 input features. `y` contains the house prices we want to predict.

### Step 5: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Splits data: **80% training** (404 rows), **20% testing** (102 rows). `random_state=42` ensures same split every time (reproducibility).

### Step 6: Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
`fit_transform` on training data: learns the mean and std of each column from training data, then scales it.
`transform` on test data: uses the SAME mean and std from training (never learn from test data — that would be data leakage).

After scaling: all features have mean ≈ 0 and std ≈ 1.

### Step 7: Build the Neural Network
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)   # No activation = linear output for regression
])
```
- **Layer 1:** 64 neurons, ReLU activation, takes 13 inputs
- **Layer 2:** 64 neurons, ReLU activation
- **Output Layer:** 1 neuron, no activation — outputs a single number (predicted price)

Total parameters: **5,121 trainable weights**

### Step 8: Compile the Model
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```
- `optimizer='adam'` — uses Adam to update weights
- `loss='mse'` — model tries to minimize Mean Squared Error during training
- `metrics=['mae']` — also tracks MAE for human-readable feedback

### Step 9: Train the Model
```python
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
```
- Trains for **100 epochs** (100 full passes over training data)
- `batch_size=32`: updates weights every 32 samples
- `validation_split=0.2`: uses 20% of training data as validation to monitor overfitting

### Step 10: Evaluate on Test Data
```python
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(test_loss)
```
- `test_loss` = MSE on unseen test data
- `test_mae` = average error in $1000s
- `rmse` = square root of MSE — same units as the target (easier to interpret)

### Step 11: Plot Predictions
```python
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
```
Each dot = one house. X-axis = actual price, Y-axis = predicted price. A perfect model would have all dots on a 45-degree diagonal line.

### Step 12: Show Sample Predictions
```python
predictions = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"Predicted: ${predictions[i]*1000:.0f} | Actual: ${y_test.iloc[i]*1000:.0f}")
```
Shows 5 actual predictions vs actual prices for comparison.

---

## How to Run

This code runs in a **Jupyter Notebook** (Google Colab or local Jupyter):

1. Open `DL_01.ipynb` in Jupyter or Google Colab
2. Upload `HousingData.csv` to the session (for Colab) or ensure it is in the same directory
3. Run all cells from top to bottom (Runtime → Run All)

---

## Output Explanation

| Metric | Value | What it Means |
|---|---|---|
| **Test MSE** | ~12.94 | Average squared error in house price prediction |
| **Test MAE** | ~2.40 | On average, the model's prediction is off by **$2,400** |
| **RMSE** | ~3.60 | Root MSE — in the same units as price ($1000s), off by ~$3,600 |

**Scatter Plot:** Points cluster near the diagonal, meaning the model is predicting well. A few outliers exist at higher prices (expensive houses are harder to predict).

**Sample Predictions:**
```
Predicted: $21,000 | Actual: $22,600
Predicted: $18,000 | Actual: $20,400
...
```
The model is reasonably close on most predictions.

---

## Conclusion

We built a 3-layer Deep Neural Network to predict house prices from 13 features. The model achieved a Test MAE of ~$2,400, meaning on average it is within $2,400 of the actual price. 

Key takeaways:
- Feature scaling (StandardScaler) was essential — without it, the model would train poorly
- No activation function at the output layer allows true regression (unbounded output)
- MSE penalizes large errors more than MAE, so we use it as the loss function

---

## Viva Questions & Answers

**Q1: What is the difference between regression and classification?**
> Regression predicts a **continuous value** (like a price or temperature). Classification predicts a **category** (like spam/not spam or A/B/C). Boston housing is regression because we predict a price ($23,500), not a class.

**Q2: Why did you use ReLU activation?**
> ReLU (Rectified Linear Unit) = max(0, x). It is simple, fast to compute, and solves the vanishing gradient problem that older activations like sigmoid caused in deep networks. Negative values become 0, positive values pass through unchanged.

**Q3: Why no activation at the output layer?**
> For regression we need the output to be any real number. Using sigmoid would limit output to (0,1), and ReLU would cut off negatives. No activation (linear) allows the model to output any value.

**Q4: What is StandardScaler and why is it used?**
> StandardScaler subtracts the mean and divides by the standard deviation of each feature. This makes all features have the same scale (mean=0, std=1). Without scaling, a feature like TAX (ranges in hundreds) would dominate over CHAS (0 or 1), and the network would struggle to learn correctly.

**Q5: What is the difference between MSE and MAE?**
> MSE = average of (actual - predicted)² — penalizes large errors more (squares them). MAE = average of |actual - predicted| — treats all errors equally. MSE is used as loss because it is differentiable (needed for gradient descent). MAE is reported because it is in the same units as the target.

**Q6: What is Adam optimizer?**
> Adam (Adaptive Moment Estimation) automatically adjusts the learning rate for each parameter using estimates of first and second moments of the gradients. It is faster and more robust than basic gradient descent.

**Q7: What is overfitting? How do you detect it?**
> Overfitting = model memorizes training data but fails on new data. You detect it by comparing training loss and validation loss: if training loss keeps decreasing but validation loss increases, the model is overfitting. We used `validation_split=0.2` to monitor this.

**Q8: What does `test_size=0.2` mean?**
> 20% of the data (about 101 rows) is reserved for testing; the remaining 80% is used for training. The test set is never seen during training — it simulates truly new, unseen data.

**Q9: Why do you use `scaler.transform()` (not `fit_transform()`) on test data?**
> We must scale test data using the SAME mean and std learned from training data. Using `fit_transform` on test data would compute different statistics from the test set — this is called **data leakage** and would give falsely good results.

**Q10: What is RMSE and how is it different from MSE?**
> RMSE = √MSE. MSE is in squared units (e.g., ($1000)²), which is hard to interpret. RMSE is in the same units as the target (e.g., $1000s), making it more intuitive. An RMSE of 3.6 means predictions are roughly $3,600 off on average.
