import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from alpaca_trade_api.rest import REST
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === Alpaca Configuration ===
API_KEY = "AKZVBC74U4Y984YA71TA"
SECRET_KEY = "k2PTshpPRhSC29LiZkCH9fUWSZjOKDGiQVnBuKqJ"
BASE_URL = "https://api.alpaca.markets"
alpaca = REST(API_KEY, SECRET_KEY, base_url=BASE_URL)

# === Streamlit UI ===
st.set_page_config(page_title="Alpaca Financial Bot", layout="wide")
st.title("📈 Alpaca Stock Price Predictor")
symbol = st.text_input("Enter Stock Symbol", "AAPL")
predict_days = st.slider("Days to Predict", 5, 60, 30)

# === Fetch Data from Alpaca ===
def fetch_alpaca_data(symbol, days=1825):  # ~5 years
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    bars = alpaca.get_bars(symbol, timeframe="1Day", start=start_str, end=end_str, feed="iex")
    df = pd.DataFrame([{
        "Date": bar.t,
        "Open": bar.o,
        "High": bar.h,
        "Low": bar.l,
        "Close": bar.c,
        "Volume": bar.v
    } for bar in bars])
    df.set_index("Date", inplace=True)
    return df

# === Predict Future Prices ===
def predict_stock_price(df, days):
    df = df.dropna()

    if len(df) < 70:
        st.warning(f"Not enough historical data. Need at least 70 days, got {len(df)}.")
        return pd.DataFrame(columns=['Date', 'Predicted_Close'])
    
    # Use OHLCV features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    sequence_length = 60
    x_data, y_data = [], []

    # Prepare sequences
    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i-sequence_length:i])
        y_data.append(scaled_data[i, features.index('Close')])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Train/validation split
    split = int(0.8 * len(x_data))
    x_train, y_train = x_data[:split], y_data[:split]
    x_val, y_val = x_data[split:], y_data[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(sequence_length, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training prediction model..."):
        history = model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            verbose=0,
            validation_data=(x_val, y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

    val_loss = model.evaluate(x_val, y_val, verbose=0)
    st.success(f"Model trained. Validation Loss: {val_loss:.6f}")

    # Start predictions
    last_sequence = scaled_data[-sequence_length:]
    current_batch = last_sequence.reshape(1, sequence_length, len(features))

    predictions = []

    for _ in range(days):
        next_pred = model.predict(current_batch, verbose=0)[0, 0]
        next_pred = np.clip(next_pred, 0, 1)
        predictions.append(next_pred)

        next_features = current_batch[0, 1:, :]
        new_row = current_batch[0, -1, :].copy()
        new_row[features.index('Close')] = next_pred
        next_features = np.vstack([next_features, new_row])

        current_batch = next_features.reshape(1, sequence_length, len(features))

    # Inverse transform predictions (only Close)
    dummy_input = np.zeros((len(predictions), len(features)))
    dummy_input[:, features.index('Close')] = predictions
    predicted_close = scaler.inverse_transform(dummy_input)[:, features.index('Close')]

    # Future dates (business days)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days, freq='B')

    return pd.DataFrame({'Date': future_dates[:len(predicted_close)], 'Predicted_Close': predicted_close.flatten()})

# === Main Logic ===
if symbol:
    try:
        with st.spinner(f"Fetching 5 years of historical data for {symbol}..."):
            df = fetch_alpaca_data(symbol, days=1825)  # Now fetching 5 years

        if df.empty:
            st.error(f"No data found for {symbol}. Please try a valid stock symbol.")
        else:
            # Basic stats
            st.subheader(f"Historical Data Overview for {symbol}")
            st.write(f"• Time Period: {df.index.min().date()} to {df.index.max().date()}")
            st.write(f"• Trading Days: {len(df)}")
            st.write(f"• Current Price: ${df['Close'].iloc[-1]:.2f}")
            st.write(f"• 30-Day Change: {(df['Close'].iloc[-1] / df['Close'].iloc[-min(30, len(df))] - 1) * 100:.2f}%")

            # Historical chart
            st.subheader(f"Historical Close Prices for {symbol}")
            st.line_chart(df['Close'])

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current", f"${df['Close'].iloc[-1]:.2f}",
                          f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f}")
            with col2:
                st.metric("30-Day High", f"${df['High'][-30:].max():.2f}")
            with col3:
                st.metric("30-Day Low", f"${df['Low'][-30:].min():.2f}")

            # Prediction
            st.subheader("Price Prediction Model")
            pred_df = predict_stock_price(df, predict_days)

            if not pred_df.empty:
                st.subheader(f"Predicted Prices for Next {predict_days} Days")

                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                historical_days = min(60, len(df))
                ax.plot(df.index[-historical_days:], df['Close'][-historical_days:], label="Historical", color="blue")
                ax.plot(pred_df['Date'], pred_df['Predicted_Close'], label="Predicted", linestyle="--", color="green")
                ax.axhline(df['Close'].iloc[-1], color='gray', linestyle=':', label="Last Close")

                last_price = df['Close'].iloc[-1]
                predicted_last = pred_df['Predicted_Close'].iloc[-1]
                percent_change = ((predicted_last - last_price) / last_price) * 100

                ax.set_title(f"{symbol} Price Forecast (Predicted {percent_change:.2f}% change in {predict_days} days)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()
                st.pyplot(fig)

                # Show prediction data
                st.subheader("Detailed Price Predictions")
                st.dataframe(pred_df.set_index('Date').style.format({"Predicted_Close": "${:.2f}"}))

                # Download button
                csv = pred_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"{symbol}_predictions.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Error: {e}")
        st.error("Stack trace:")
        st.exception(e)
 