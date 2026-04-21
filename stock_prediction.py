# Stock Price Prediction using LSTM Neural Network
# Predicts the closing stock price of Apple Inc. (AAPL)
# using the past 60 days of stock price data

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# ─── 1. DATA COLLECTION ───────────────────────────────────────────────────────

# Fetch historical stock data for Apple Inc. from Yahoo Finance
df = yf.download('AAPL', start='2012-01-01', end='2024-03-16')
df.columns = df.columns.get_level_values(0)
df = df[['Close']]
print(f"Dataset shape: {df.shape}")

# Visualise closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# ─── 2. DATA PREPROCESSING ────────────────────────────────────────────────────

# Filter to closing price only and convert to numpy array
data = df
dataset = data.values

# Use 80% of data for training
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale data to range [0, 1] for LSTM input
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# ─── 3. TRAINING DATA PREPARATION ────────────────────────────────────────────

train_data = scaled_data[0:training_data_len, :]

# Build sequences: each input (X) is 60 days, each label (y) is the next day's price
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays and reshape for LSTM input (samples, timesteps, features)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ─── 4. MODEL ARCHITECTURE ────────────────────────────────────────────────────

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile with Adam optimiser and MSE loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (epochs=1 for demonstration; increase for better accuracy)
model.fit(x_train, y_train, batch_size=1, epochs=1)

# ─── 5. TESTING & EVALUATION ──────────────────────────────────────────────────

# Prepare test data (include 60-day lookback window)
test_data = scaled_data[training_data_len - 60:, :]

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert and reshape for LSTM input
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Generate predictions and reverse scaling
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate and display RMSE
rmse = np.sqrt(((predictions - y_test) ** 2).mean())
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# ─── 6. VISUALISATION ─────────────────────────────────────────────────────────

train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('LSTM Stock Price Prediction vs Actual')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# ─── 7. SINGLE DAY PREDICTION ─────────────────────────────────────────────────

# Predict the closing price using the most recent 60 days
apple_quote = yf.download('AAPL', start='2012-01-01', end='2024-03-16')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values

# Scale, reshape, and predict
last_60_days_scaled = scaler.transform(last_60_days)
X_test = np.array([last_60_days_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(f"Predicted closing price: ${pred_price[0][0]:.2f}")
