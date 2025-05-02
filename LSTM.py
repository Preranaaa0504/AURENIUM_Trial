# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from datetime import datetime

# Set date range for data
start = '2010-01-01'
end = '2025-04-28'

# Fetch stock data using yfinance
df = yf.download('TSLA', start=start, end=end)

# Reset index to convert Date from index to column
df = df.reset_index()

# Calculate moving averages (optional analysis)
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Split data into training and testing (70% training, 30% testing)
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data: use 100 previous data points to predict the next one
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train for LSTM [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50)

# Save the trained model
model.save('my_model.h5')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape x_test for LSTM
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Make predictions
y_pred = model.predict(x_test)

# Inverse scaling to get actual values (approximation with fixed factor)
scale_factor = 1 / 0.00489511
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('TSLA Stock Price')
plt.title('TSLA Price Prediction using LSTM')
plt.legend()
plt.grid(True)
plt.show()
