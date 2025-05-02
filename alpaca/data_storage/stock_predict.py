import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load prices from CSV
df = pd.read_csv('alpaca/data_storage/prices.csv', parse_dates=['timestamp'])
symbol = 'NVDA'  # Change as needed
df = df[df['symbol'] == symbol].sort_values('timestamp')

# Use OHLCV data
features = ['open', 'high', 'low', 'close', 'volume']
data = df[features].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare sequences for LSTM (multivariate)
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step, 3])  # Predict 'close' price
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Train/test split (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model (multivariate input)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# Predict next 7 days
last_sequence = scaled_data[-time_step:]
future_predictions = []

# We'll keep the full OHLCV for each predicted step
for _ in range(7):
    input_seq = last_sequence.reshape(1, time_step, len(features))
    pred_scaled_close = model.predict(input_seq, verbose=0)[0][0]
    
    # Create new entry: we only predict close, so we keep other features same as last step
    new_entry = last_sequence[-1].copy()
    new_entry[3] = pred_scaled_close  # update 'close'
    
    # Optionally adjust Open, High, Low if you have better logic (here we keep them same)
    last_sequence = np.vstack([last_sequence[1:], new_entry])
    
    future_predictions.append(pred_scaled_close)

# Inverse scale to get real prices (close only)
predicted_close_prices = []
for pred_scaled in future_predictions:
    dummy_row = np.zeros((1, len(features)))
    dummy_row[0, 3] = pred_scaled  # put pred in 'close' column
    inv_scaled = scaler.inverse_transform(dummy_row)
    predicted_close_prices.append(inv_scaled[0, 3])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['close'], label='Historical Close Price')
future_dates = pd.date_range(df['timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=7)
plt.plot(future_dates, predicted_close_prices, label='Predicted Close Prices', marker='o', linestyle='--', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'7-Day Future Close Price Prediction for {symbol}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
