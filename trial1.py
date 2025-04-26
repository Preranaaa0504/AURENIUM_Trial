import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import yfinance as yf
import streamlit as st
import os
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import json
from pathlib import Path

# API Configuration
ALPHA_VANTAGE_API_KEY = 'N81EYEWKOSM8RY8K'
FMP_API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

# Constants
MODEL_PATH = Path("models")
DATA_PATH = Path("data")

# Create directories if they don't exist
MODEL_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)

# Main App
st.set_page_config(page_title="Stock ChatBot", layout="wide")
st.title("Stock Analysis & Prediction ChatBot")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "tickers_info" not in st.session_state:
    st.session_state.tickers_info = {}

# Helper functions
def extract_tickers(text):
    standard_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    all_tickers = list(set(standard_tickers + dollar_tickers))
    common_words = ["I", "A", "FOR", "IN", "ON", "AT", "BY", "AND", "OR", "THE", "TO", "ME", "HOW", "WHAT", "WHEN", "WHY", "IS", "ARE", "WILL"]
    return [ticker for ticker in all_tickers if ticker not in common_words]

def extract_time_period(text):
    text = text.lower()
    if any(term in text for term in ["1 day", "today", "24 hour"]):
        return "1d"
    elif any(term in text for term in ["5 day", "week", "weekly"]):
        return "5d"
    elif any(term in text for term in ["1 month", "monthly", "30 day"]):
        return "1mo"
    elif any(term in text for term in ["3 month", "quarter", "quarterly"]):
        return "3mo"
    elif any(term in text for term in ["6 month", "half year"]):
        return "6mo"
    elif any(term in text for term in ["1 year", "annual", "yearly"]):
        return "1y"
    elif any(term in text for term in ["2 year", "2 years"]):
        return "2y"
    elif any(term in text for term in ["5 year", "long term"]):
        return "5y"
    elif any(term in text for term in ["10 year", "decade"]):
        return "10y"
    elif any(term in text for term in ["max", "all time", "inception"]):
        return "max"
    return "1y"  # Default

def extract_prediction_days(text):
    text = text.lower()
    days_pattern = r'(\d+)\s*(day|days)'
    weeks_pattern = r'(\d+)\s*(week|weeks)'
    months_pattern = r'(\d+)\s*(month|months)'
    
    days_match = re.search(days_pattern, text)
    weeks_match = re.search(weeks_pattern, text)
    months_match = re.search(months_pattern, text)
    
    if days_match:
        return int(days_match.group(1))
    elif weeks_match:
        return int(weeks_match.group(1)) * 7
    elif months_match:
        return int(months_match.group(1)) * 30
    elif "tomorrow" in text:
        return 1
    elif "next week" in text:
        return 7
    elif "next month" in text:
        return 30
    else:
        return 30  # Default 30 days

def get_stock_data(ticker, period='5y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        
        # Convert to common column names
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return data
    except Exception as e:
        st.error(f"Error retrieving stock data: {e}")
        return None

def prepare_stock_data(data, sequence_length=60):
    data.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data['close'].values.reshape(-1,1))

    X, Y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        Y.append(data_scaled[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    return X_train, X_test, Y_train, Y_test, scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, Y_train):
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history

def generate_predictions(model, data, scaler, sequence_length=60, days_to_predict=30):
    closing_prices = data['close'].values
    scaled_data = scaler.transform(closing_prices.reshape(-1, 1))

    predicted_prices = []
    current_batch = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    for i in range(days_to_predict):
        next_prediction = model.predict(current_batch, verbose=0)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

    return predicted_prices

def get_stock_prediction(ticker, days_to_predict=30):
    """Get stock prediction, training model if needed"""
    model_file = MODEL_PATH / f"{ticker}_model.h5"
    scaler_file = MODEL_PATH / f"{ticker}_scaler.pkl"
    
    # Check if we already have a model and recent predictions
    if ticker in st.session_state.predictions and len(st.session_state.predictions[ticker]["dates"]) >= days_to_predict:
        # We already have predictions for this ticker
        return st.session_state.predictions[ticker]
    
    # Get the stock data
    with st.spinner(f"Analyzing {ticker} stock data..."):
        stock_data = get_stock_data(ticker, period='5y')
        
        if stock_data is None or len(stock_data) < 100:  # Need enough data
            return {"error": f"Not enough historical data for {ticker}"}
        
        # Check if we have a saved model
        if model_file.exists() and scaler_file.exists():
            model = load_model(model_file)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # Train a new model
            X_train, X_test, Y_train, Y_test, scaler = prepare_stock_data(stock_data)
            model, history = train_model(X_train, Y_train)
            
            # Save the model and scaler
            model.save(model_file)
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Generate predictions
        predicted_prices = generate_predictions(model, stock_data, scaler, days_to_predict=days_to_predict)
        
        # Get latest price
        latest_price = stock_data['close'].iloc[-1]
        
        # Generate future dates
        last_date = stock_data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=days_to_predict+1)[1:]
        
        # Calculate potential return
        potential_return = ((predicted_prices[-1] - latest_price) / latest_price) * 100
        
        # Calculate trend direction
        is_uptrend = predicted_prices[-1] > latest_price
        
        # Calculate past performance for context
        past_month_return = ((latest_price - stock_data['close'].iloc[-21]) / stock_data['close'].iloc[-21]) * 100 if len(stock_data) > 21 else 0
        past_year_return = ((latest_price - stock_data['close'].iloc[-252]) / stock_data['close'].iloc[-252]) * 100 if len(stock_data) > 252 else 0
        
        # Store predictions
        predictions = {
            "ticker": ticker,
            "latest_price": latest_price,
            "predicted_prices": predicted_prices,
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "potential_return": potential_return,
            "is_uptrend": is_uptrend,
            "past_month_return": past_month_return,
            "past_year_return": past_year_return
        }
        
        st.session_state.predictions[ticker] = predictions
        return predictions

def plot_prediction(prediction):
    ticker = prediction["ticker"]
    dates = pd.to_datetime(prediction["dates"])
    predicted_prices = prediction["predicted_prices"]
    latest_price = prediction["latest_price"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual price point
    ax.scatter(dates[0] - pd.Timedelta(days=1), latest_price, color='blue', s=50, label='Current Price')
    
    # Plot predicted prices
    ax.plot(dates, predicted_prices, 'r-o', label=f'Predicted Prices', markersize=4)
    
    # Fill area based on trend
    if prediction["is_uptrend"]:
        ax.fill_between(dates, latest_price, predicted_prices, color='green', alpha=0.2)
    else:
        ax.fill_between(dates, latest_price, predicted_prices, color='red', alpha=0.2)
    
    # Add labels
    ax.set_title(f'{ticker} Stock Price Prediction', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate the final predicted price
    ax.annotate(f'${predicted_prices[-1]:.2f}',
                xy=(dates[-1], predicted_prices[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=12, fontweight='bold',
                color='red')
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    return fig

def get_company_info(ticker):
    """Get company information from Yahoo Finance"""
    if ticker in st.session_state.tickers_info:
        return st.session_state.tickers_info[ticker]
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract the most relevant information
        company_info = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "price": info.get("currentPrice", 0),
            "beta": info.get("beta", 0),
            "description": info.get("longBusinessSummary", "No description available.")
        }
        
        st.session_state.tickers_info[ticker] = company_info
        return company_info
    except Exception as e:
        return {"name": ticker, "error": str(e)}

def plot_price_trend(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data["Close"], label=f"{ticker} Close Price", linewidth=2)
        
        # Calculate percent change
        percent_change = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
        color = 'green' if percent_change >= 0 else 'red'
        
        ax.set_title(f"{ticker} Price Trend - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.annotate(f"{percent_change:.2f}%", xy=(data.index[-1], data["Close"].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points", color=color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Summary stats
        summary = {
            "ticker": ticker,
            "start_price": data['Close'].iloc[0],
            "end_price": data['Close'].iloc[-1],
            "percent_change": percent_change,
            "period": period
        }
        
        return fig, summary
    except Exception as e:
        return None, {"error": str(e)}

def analyze_query(query):
    """Analyze the user's query and determine what action to take"""
    query_lower = query.lower()
    
    # Extract tickers from the query
    tickers = extract_tickers(query)
    
    # Check if this is a price prediction query
    if any(term in query_lower for term in ["predict", "forecast", "future", "price", "will", "trend", "go up", "go down"]):
        return {
            "type": "prediction",
            "tickers": tickers,
            "days": extract_prediction_days(query_lower)
        }
    
    # Check if this is a price trend query
    elif any(term in query_lower for term in ["trend", "chart", "performance", "historical", "how has", "compare"]):
        return {
            "type": "trend",
            "tickers": tickers,
            "period": extract_time_period(query_lower)
        }
    
    # Check if this is an information query
    elif any(term in query_lower for term in ["info", "about", "company", "who is", "what is", "tell me about"]):
        return {
            "type": "info",
            "tickers": tickers
        }
    
    # General query - try to understand based on tickers
    elif tickers:
        return {
            "type": "general",
            "tickers": tickers
        }
    
    # Fallback
    else:
        return {
            "type": "unknown",
            "tickers": []
        }

def generate_response(query_analysis):
    """Generate a response based on the query analysis"""
    query_type = query_analysis["type"]
    tickers = query_analysis["tickers"]
    
    if query_type == "prediction":
        if not tickers:
            return "I need a specific ticker symbol to predict stock prices. Please mention a stock ticker in your query."
        
        responses = []
        for ticker in tickers[:3]:  # Limit to 3 tickers
            prediction = get_stock_prediction(ticker, query_analysis["days"])
            
            if "error" in prediction:
                responses.append(f"Sorry, I couldn't generate a prediction for {ticker}: {prediction['error']}")
                continue
            
            # Create a response
            trend_direction = "increase" if prediction["is_uptrend"] else "decrease"
            confidence_level = "high" if abs(prediction["potential_return"]) > 10 else "moderate"
            
            response = f"Based on my analysis of {ticker}, I predict the price will {trend_direction} by approximately {prediction['potential_return']:.2f}% over the next {len(prediction['predicted_prices'])} days."
            response += f" The current price is ${prediction['latest_price']:.2f} and my prediction for {prediction['dates'][-1]} is ${prediction['predicted_prices'][-1]:.2f}."
            
            # Add context from past performance
            if abs(prediction["past_month_return"]) > 10:
                response += f" For context, {ticker} has {('increased' if prediction['past_month_return'] > 0 else 'decreased')} by {abs(prediction['past_month_return']):.2f}% in the past month."
            
            # Add confidence statement
            response += f" I have {confidence_level} confidence in this prediction based on historical patterns."
            
            responses.append(response)
        
        return "\n\n".join(responses)
    
    elif query_type == "trend":
        if not tickers:
            return "I need a specific ticker symbol to analyze price trends. Please mention a stock ticker in your query."
        
        responses = []
        for ticker in tickers[:3]:  # Limit to 3 tickers
            fig, summary = plot_price_trend(ticker, query_analysis["period"])
            
            if fig is None or "error" in summary:
                responses.append(f"Sorry, I couldn't analyze the trend for {ticker}.")
                continue
            
            performance = "up" if summary["percent_change"] > 0 else "down"
            
            response = f"{ticker} has gone {performance} {abs(summary['percent_change']):.2f}% over the {summary['period']} period."
            response += f" The price started at ${summary['start_price']:.2f} and is currently at ${summary['end_price']:.2f}."
            
            responses.append(response)
        
        return "\n\n".join(responses)
    
    elif query_type == "info":
        if not tickers:
            return "I need a specific ticker symbol to provide company information. Please mention a stock ticker in your query."
        
        responses = []
        for ticker in tickers[:3]:  # Limit to 3 tickers
            info = get_company_info(ticker)
            
            if "error" in info:
                responses.append(f"Sorry, I couldn't find information for {ticker}.")
                continue
            
            response = f"{info['name']} ({ticker}) is a company in the {info['sector']} sector, specifically in the {info['industry']} industry."
            
            if info['market_cap'] > 0:
                market_cap_billions = info['market_cap'] / 1_000_000_000
                response += f" It has a market capitalization of ${market_cap_billions:.2f} billion."
            
            if info['pe_ratio'] > 0:
                response += f" The trailing P/E ratio is {info['pe_ratio']:.2f}."
            
            if info['dividend_yield'] > 0:
                dividend_percent = info['dividend_yield'] * 100
                response += f" The dividend yield is {dividend_percent:.2f}%."
            
            response += f" The current stock price is ${info['price']:.2f}."
            
            # Add a brief company description
            if len(info['description']) > 50:
                first_sentence = info['description'].split('.')[0] + "."
                response += f"\n\nBrief description: {first_sentence}"
            
            responses.append(response)
        
        return "\n\n".join(responses)
    
    elif query_type == "general":
        if not tickers:
            return "I'm a stock analysis chatbot. I can help you with stock price predictions, historical trends, and company information. Please ask me about specific stocks by mentioning their ticker symbols."
        
        # For general queries with tickers, provide a summary of the stocks
        responses = []
        for ticker in tickers[:3]:  # Limit to 3 tickers
            info = get_company_info(ticker)
            
            if "error" in info:
                responses.append(f"Sorry, I couldn't find information for {ticker}.")
                continue
            
            fig, summary = plot_price_trend(ticker, "1mo")
            
            response = f"{info['name']} ({ticker}) is currently trading at ${info['price']:.2f}."
            
            if fig is not None and "error" not in summary:
                performance = "up" if summary["percent_change"] > 0 else "down"
                response += f" It has gone {performance} {abs(summary['percent_change']):.2f}% in the past month."
            
            responses.append(response)
        
        return "\n\n".join(responses)
    
    else:  # Unknown query type
        return "I'm a stock analysis chatbot. I can help you with stock price predictions, historical trends, and company information. Please ask me about specific stocks by mentioning their ticker symbols."

# UI Elements
st.markdown("""
### Welcome to the Stock Analysis & Prediction ChatBot

I can help you with:
- Predicting future stock prices
- Analyzing historical price trends
- Providing company information
- Comparing multiple stocks

Just ask me anything about stocks using their ticker symbols (e.g., AAPL, MSFT, GOOG).
""")

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f"👤 **You**: {message['content']}")
    else:
        st.markdown(f"🤖 **Bot**: {message['content']}")
        
        # Display charts if available
        if "charts" in message:
            for chart in message["charts"]:
                st.pyplot(chart)

# User input
user_query = st.text_input("Ask me about stocks:", key="user_input")

if st.button("Send") or user_query:
    if user_query:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_query})
        
        # Analyze query
        query_analysis = analyze_query(user_query)
        
        # Generate response
        bot_response = generate_response(query_analysis)
        
        # Prepare charts if needed
        charts = []
        if query_analysis["type"] == "prediction" and query_analysis["tickers"]:
            for ticker in query_analysis["tickers"][:3]:  # Limit to 3 tickers
                if ticker in st.session_state.predictions:
                    fig = plot_prediction(st.session_state.predictions[ticker])
                    charts.append(fig)
        
        elif query_analysis["type"] == "trend" and query_analysis["tickers"]:
            for ticker in query_analysis["tickers"][:3]:  # Limit to 3 tickers
                fig, _ = plot_price_trend(ticker, query_analysis["period"])
                if fig:
                    charts.append(fig)
        
        # Add bot response to conversation
        st.session_state.conversation.append({
            "role": "assistant", 
            "content": bot_response,
            "charts": charts
        })
        
        # Clear the input field
        st.text_input("Ask me about stocks:", value="", key="user_input_clear")
        
        # Rerun the app to display the new messages
        st.rerun()

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    st.rerun()