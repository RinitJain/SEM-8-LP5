# Practical 4: Google Stock Price Prediction using RNN and LSTM

## Objective

Predict Google's stock closing price using historical time series data. We implement and compare two models: a **SimpleRNN** (basic Recurrent Neural Network) and an **LSTM** (Long Short-Term Memory). Both are designed for sequential data where the order of values matters. We use 4 consecutive days of prices to predict the next day's price.

---

## Theory / Background

### What is a Time Series?
A time series is a sequence of values recorded over time — like stock prices, temperature readings, or sensor data. The key property is that **order matters**: today's price depends on yesterday's price, which depends on the day before, and so on.

### Why Can't We Use a Regular Neural Network?
Regular Dense networks process each input independently — they have no memory of previous inputs. But for stock prices, the recent trend is crucial. We need a network that **remembers past values** while processing the current one.

### What is an RNN?
A Recurrent Neural Network (RNN) processes sequences by maintaining a **hidden state** — a kind of memory that gets updated at each time step. When it processes day 3's price, it also has access to a hidden state that was updated from days 1 and 2.

**Simple Formula:**
```
hidden_state_t = tanh(W_h × hidden_state_{t-1} + W_x × input_t + bias)
output_t = W_o × hidden_state_t
```
The hidden state carries information from previous time steps into the current computation.

### What is the Vanishing Gradient Problem?
When training RNNs with many time steps (long sequences), gradients shrink exponentially as they travel back through time during backpropagation. After many steps, the gradient becomes so tiny that early time steps stop learning. This is the **vanishing gradient problem** — SimpleRNN suffers from it, meaning it forgets events from long ago.

### What is LSTM?
LSTM (Long Short-Term Memory) was designed to solve the vanishing gradient problem. It adds a separate **cell state** (long-term memory) alongside the hidden state (short-term memory), controlled by three **gates**:

| Gate | Purpose |
|---|---|
| **Forget Gate** | Decides what information from the old cell state to erase/forget |
| **Input Gate** | Decides what new information to write to the cell state |
| **Output Gate** | Decides what part of the cell state to output as the hidden state |

These gates use sigmoid activations (output 0–1): 0 = completely block, 1 = completely allow. By selectively remembering and forgetting, LSTM can maintain relevant information over long sequences.

### SimpleRNN vs LSTM:
| Aspect | SimpleRNN | LSTM |
|---|---|---|
| Memory | Short-term only | Short-term + Long-term |
| Vanishing gradient | Yes, suffers from it | Largely solved |
| Complexity | Simpler, fewer parameters | More complex, more parameters |
| Performance on long sequences | Poor | Much better |

### Key Terms:
| Term | Plain English Meaning |
|---|---|
| **MinMaxScaler** | Scales values to the 0–1 range using min and max. Good for time series. |
| **Sequence window (step)** | Group of consecutive past values used to predict the next value. Here step=4 means "use 4 days to predict the 5th" |
| **return_sequences=True** | When True, the RNN/LSTM outputs a hidden state at every time step (needed when stacking RNN layers). When False (default), only the final hidden state is output. |
| **RMSE** | Root Mean Squared Error — measures prediction error in the same units as the target (stock price $) |
| **Dropout** | Randomly zeros some outputs during training to prevent overfitting |

---

## Dataset

**File:** `goog.csv`

**Size:** 61 rows × 6 columns

**Source:** Google (Alphabet Inc.) stock price data

| Column | Meaning |
|---|---|
| Date | Date of the trading day |
| Open | Price at market opening |
| High | Highest price during the day |
| Low | Lowest price during the day |
| **Close** | **Price at market close — the one we predict** |
| Volume | Number of shares traded |

We only use the **Close** price column. The dataset is small (61 records), making it a simple demonstration dataset.

---

## Step-by-Step Code Walkthrough

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
```
- `MinMaxScaler` — scales price data to 0–1 range
- `SimpleRNN`, `LSTM` — the two recurrent layer types we'll compare
- `mean_squared_error` — for computing RMSE

### Step 2: Load and Extract Close Price
```python
df = pd.read_csv('goog.csv')
data = df['Close'].values.reshape(-1, 1)
```
Read the CSV and extract only the 'Close' column. `.reshape(-1, 1)` converts the 1D array (shape: 61,) to 2D (shape: 61, 1) because MinMaxScaler requires 2D input.

### Step 3: Normalize Data
```python
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
```
MinMaxScaler transforms values to the 0–1 range:
`scaled_value = (value - min) / (max - min)`

Why MinMaxScaler (not StandardScaler)?
- Stock prices have a natural bounded range and no assumed Gaussian distribution
- MinMaxScaler is better for bounded sequences; StandardScaler assumes normally distributed data
- We need to **inverse transform** predictions later to get actual prices back

### Step 4: Train-Test Split
```python
train_size = int(len(data) * 0.8)   # 48 rows for training
train = data[:train_size]
test = data[train_size:]              # 13 rows for testing
```
First 80% (48 data points) = training. Last 20% (13 data points) = testing.
Note: For time series, we do NOT shuffle — we must keep temporal order.

### Step 5: Create Sequence Windows
```python
def create_dataset(data, step=4):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step])   # 4 consecutive days
        y.append(data[i])           # the value at day i (target)
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train)
X_test, y_test = create_dataset(test)
```
This function creates overlapping windows:
- Window 1: days [0,1,2,3] → predict day 0
- Window 2: days [1,2,3,4] → predict day 1
- ...

`X` shape becomes: (samples, 4, 1) — 4 time steps, 1 feature (price)

### Step 6: Plot Raw Stock Price
```python
plt.plot(data)
plt.title("Google Stock Price Over Time")
```
Visualizes the normalized stock price over all 61 time points.

### Step 7: Build SimpleRNN Model
```python
model_rnn = Sequential()
model_rnn.add(SimpleRNN(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_rnn.add(Dropout(0.2))
model_rnn.add(SimpleRNN(32))
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(1))
```
- **First SimpleRNN (64 units):** `return_sequences=True` — passes hidden state at EACH time step to the next layer (needed when stacking RNN layers)
- **Dropout(0.2):** Drops 20% of neurons randomly during training
- **Second SimpleRNN (32 units):** `return_sequences=False` (default) — only outputs the final hidden state
- **Dropout(0.2):** Another dropout layer
- **Dense(1):** Outputs a single predicted price value

### Step 8: Train SimpleRNN
```python
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
history_rnn = model_rnn.fit(X_train, y_train, epochs=30)
```
Trained for 30 epochs. Loss function: MSE (measures how far predictions are from actual values in squared terms).

### Step 9: Predict and Inverse Transform (RNN)
```python
y_pred_rnn = model_rnn.predict(X_test)
y_pred_rnn = scaler.inverse_transform(y_pred_rnn)  # Convert back to actual price
y_test_inv = scaler.inverse_transform(y_test)       # Convert actual values back too
```
`inverse_transform` reverses the MinMaxScaler normalization to get back actual dollar prices. Both predictions and actuals must be inverse-transformed for a fair comparison.

### Step 10: Plot RNN Predictions
```python
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_rnn, label='RNN Predicted')
```
Overlays actual vs predicted stock prices for the test set.

### Step 11: Build LSTM Model
```python
model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
```
Same architecture as RNN but using **LSTM layers** instead of SimpleRNN layers. LSTM internally manages forget, input, and output gates.

### Step 12: Train LSTM
```python
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
history_lstm = model_lstm.fit(X_train, y_train, epochs=50)
```
LSTM trained for 50 epochs (more than RNN because LSTM has more parameters to optimize).

### Step 13: Predict with LSTM
```python
y_pred_lstm = model_lstm.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
```

### Step 14: Compare Both Models
```python
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_rnn, label='RNN')
plt.plot(y_pred_lstm, label='LSTM')
plt.title("RNN vs LSTM Comparison")
```
Side-by-side comparison of both models against the actual prices.

### Step 15: Compute RMSE
```python
rmse_rnn = np.sqrt(mean_squared_error(y_test_inv, y_pred_rnn))
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm))
print("RNN RMSE:", rmse_rnn)
print("LSTM RMSE:", rmse_lstm)
```
RMSE = √(average of squared errors). Smaller RMSE = better predictions.

---

## How to Run

1. Open `DL_04.ipynb` in Jupyter or Google Colab
2. Upload `goog.csv` to the session directory
3. Run all cells top to bottom

---

## Output Explanation

| Model | RMSE | Interpretation |
|---|---|---|
| **SimpleRNN** | ~1.247 | Avg prediction error of ~$1.25 per share |
| **LSTM** | ~3.957 | Avg prediction error of ~$3.96 per share |

**Surprising result:** RNN outperformed LSTM here. Why? The dataset is very small (61 records → only ~9–10 test points). LSTM has many more parameters and can overfit small datasets. With more data and longer sequences, LSTM would generally outperform SimpleRNN.

**Actual vs Predicted Plots:**
Both models trace the general trend of the stock price, but predictions lag slightly behind the actual values. This is expected behavior for time series prediction — the model is learning historical patterns.

**RNN vs LSTM Comparison Plot:**
Shows three overlapping lines: actual price (ground truth), RNN prediction, and LSTM prediction. The RNN line tracks closer to the actual price than LSTM on this small dataset.

---

## Conclusion

We implemented both SimpleRNN and LSTM to predict Google stock prices using a window of 4 days. SimpleRNN achieved lower RMSE (1.247) than LSTM (3.957) on this small dataset, which is counterintuitive but explainable by overfitting in LSTM due to the tiny dataset size.

Key takeaways:
- RNNs are designed for sequential data — they maintain state across time steps
- LSTM solves the vanishing gradient problem of SimpleRNN using gates, and outperforms SimpleRNN on long sequences with more data
- MinMaxScaler is essential — both for stable training and for getting back real prices through inverse_transform
- Time series data must NOT be shuffled — temporal order must be preserved

---

## Viva Questions & Answers

**Q1: What is an RNN and how is it different from a regular neural network?**
> A Recurrent Neural Network processes sequential data by maintaining a hidden state — a memory — that is updated at each time step and passed to the next. A regular Dense network processes each input independently with no memory of past inputs. RNNs are essential for time series, text, and speech data where order matters.

**Q2: What is the vanishing gradient problem?**
> During backpropagation through time in long sequences, gradients are multiplied repeatedly. Each multiplication shrinks the gradient (since weights are typically < 1). After many time steps, gradients become nearly zero — early time steps stop learning. This means SimpleRNN forgets events from the distant past.

**Q3: What is LSTM and how does it solve the vanishing gradient problem?**
> LSTM (Long Short-Term Memory) adds a separate cell state — a "long-term memory highway" — that can carry information across many time steps without repeated multiplication. Three gates (forget, input, output) control what is written, erased, or read from this cell state. The additive updates prevent gradients from vanishing.

**Q4: What are the three gates in LSTM?**
> 1. **Forget Gate:** Decides what to erase from long-term memory (cell state). Multiplied by current cell state — values near 0 erase, values near 1 keep.
> 2. **Input Gate:** Decides what new information to add to cell state from current input.
> 3. **Output Gate:** Decides what part of the cell state to expose as the hidden state (short-term output).
> All gates use sigmoid activation (output 0–1).

**Q5: Why use MinMaxScaler instead of StandardScaler for stock data?**
> MinMaxScaler scales values to [0, 1] using min and max. Stock prices are non-negative and have a natural range. StandardScaler assumes data is normally distributed (bell curve), which stock prices are not. Also, MinMaxScaler's inverse_transform is simpler and more intuitive for converting predictions back to real prices.

**Q6: What is `return_sequences=True` and when is it used?**
> By default, an RNN/LSTM only outputs the hidden state from the FINAL time step. When `return_sequences=True`, it outputs the hidden state at EVERY time step. This is needed when stacking two RNN/LSTM layers — the second layer needs a sequence (one vector per time step) as input, not just the final vector.

**Q7: What is RMSE and why do we use it for evaluation?**
> RMSE = √(mean of squared differences between actual and predicted values). It is in the same units as the prediction (here: stock price in $). Squaring penalizes large errors more. A lower RMSE means the model's predictions are closer to actual values.

**Q8: Why did SimpleRNN outperform LSTM in this experiment?**
> The dataset is very small (only 61 records → ~9 test predictions). LSTM has significantly more parameters (forget gate, input gate, output gate, cell state) than SimpleRNN. With little data, LSTM overfits more easily. With larger datasets and longer sequences, LSTM would typically outperform SimpleRNN.

**Q9: What does `create_dataset(data, step=4)` do?**
> It creates input-output pairs for supervised learning from the time series. With step=4, each input sample X contains 4 consecutive price values, and y is the target value. This is called a "sliding window" approach — the window slides one step at a time through the data.

**Q10: Why do we NOT shuffle data for time series?**
> Time series data has temporal dependencies — the order of data points matters. Shuffling would destroy this order. For example, day 5's data must come after days 1–4 in training, not randomly. Shuffling would cause data leakage (future data leaking into training sequences).

**Q11: What does `scaler.inverse_transform()` do and why is it necessary?**
> `inverse_transform` reverses the MinMaxScaler normalization, converting scaled values back to the original price scale. We need this because: (1) we train on scaled (0–1) data, (2) the model outputs scaled predictions (0–1 range), and (3) we need real dollar prices to compute a meaningful RMSE and to plot interpretable results.

**Q12: What is the difference between RNN and LSTM in terms of architecture?**
> SimpleRNN has one hidden state (short-term memory only). LSTM has two states: hidden state (short-term) and cell state (long-term). LSTM uses 4 neural network layers internally (3 gates + candidate cell update), while SimpleRNN uses just 1. LSTM has roughly 4× more parameters than a SimpleRNN of the same size.
