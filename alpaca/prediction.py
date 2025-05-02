import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model   
from sklearn.preprocessing import MinMaxScaler

# Set start and end dates
start = '2010-01-01'
end = '2025-3-31'

st.title('Stock Trend Prediction')
user_input = st.text_input("Enter Stock Ticker", 'AAPL')
df = yf.download(user_input, start=start, end=end)  # Download stock data

# Describing Data
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('keras_model.h5')

# Prepare data for predictions
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True) 
input_data = scaler.fit_transform(final_df)

# Prepare input for prediction (use past 100 days)
x_input = input_data[-100:].reshape(1, -1)

# Get user input for the number of days to predict
prediction_days = st.number_input("Enter number of days to predict", min_value=1, max_value=365, value=30)

# Predict the next 'prediction_days' days
predicted_prices = []
for _ in range(prediction_days):
    prediction = model.predict(x_input)
    predicted_prices.append(prediction[0, 0])
    x_input = np.roll(x_input, -1)  # Shift input
    x_input[0, -1] = prediction  # Append predicted value

# Inverse scaling of the predictions
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plot the predictions
st.subheader(f'Predicted Closing Prices for {user_input} for the Next {prediction_days} Days')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, 'r', label='Predicted')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Display predicted prices in a table
predicted_df = pd.DataFrame({'Day': np.arange(1, prediction_days + 1), 'Predicted Price': predicted_prices.flatten()})
st.subheader(f'Predicted Prices for the Next {prediction_days} Days')
st.dataframe(predicted_df)
