import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Define the stock symbol and date range
start_date = "2010-01-01"
end_date = "2024-08-03"

st.title("Stock Trend Prediction")

# User input for stock ticker
user_input = st.text_input("Enter Stock Ticker", "PYPL")
symbol = user_input.upper()

# Fetch data
df = yf.download(symbol, start=start_date, end=end_date)

st.subheader("Data from 2010-2024")
st.write(df.describe())

# Visualizations
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Forecast
y_pred = model.predict(X_test)

y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
date_range = pd.date_range(start=start_date, periods=len(df), freq='B')

# Plot
st.subheader("Actual vs Predicted Closing Price")
fig = plt.figure(figsize=(12,6))
plt.plot(date_range[-len(y_test):], y_test, label='True')
plt.plot(date_range[-len(y_test):], y_pred, label='Predicted', color='red')
plt.legend()
st.pyplot(fig)
