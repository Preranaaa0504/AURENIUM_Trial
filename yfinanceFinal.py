import yfinance as yf
import requests
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing import TypedDict, Sequence
from bs4 import BeautifulSoup
from nselib import capital_market
from nselib import derivatives
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model

# API Configuration
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'O6dpb7fj0SdWc42Rwb07rZjxqUbXnjD6'

# Add Alpha Vantage API key to your environment variables
#os.environ["ALPHA_VANTAGE_API_KEY"] = "0UAIZ53Q3FCCTY9U"

# LangSmith setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

# Set environment variables for OpenRouter

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize DeepSeek AI via OpenRouter
model = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
)

# os.environ["GROK_API_KEY"] = "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"
# os.environ["GROK_API_BASE"] = "https://api.groq.com/openai/v1"

# # Initialize Grok model
# model = ChatOpenAI(
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     openai_api_key=os.environ["GROK_API_KEY"],
#     openai_api_base=os.environ["GROK_API_BASE"],
# )

# Define State
class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str
    original_question: str

# System prompt with visualization awareness
system_prompt = """
You are a financial risk assessment assistant. Provide investment advice based on financial information.
When users request visualizations or when visual analysis would be helpful, acknowledge that you'll 
generate appropriate charts and explain what they'll show. The system will automatically create 
visualizations based on the query. Provide brief analysis alongside any visualization.
For price predictions, acknowledge that you'll generate LSTM-based forecasts, including real-time predictions when requested.
"""

# LangGraph setup
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

def call_model(state: State):
    prompt = prompt_template.invoke({"messages": state["messages"], "language": state.get("language", "English")})
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")
app = workflow.compile()

# Streamlit UI setup
st.set_page_config(page_title="Financial Chatbot", layout="wide")
st.title("Financial Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []
if "original_question" not in st.session_state:
    st.session_state.original_question = None
if "message_charts" not in st.session_state:
    st.session_state.message_charts = {}

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask about stocks or request visualizations:")

# Helper functions
def extract_tickers(query):
    standard_tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', query)
    all_tickers = list(set(standard_tickers + dollar_tickers))
    common_words = ["I", "A", "FOR", "IN", "ON", "AT", "BY", "AND", "OR", "THE", "TO"]
    return [ticker for ticker in all_tickers if ticker not in common_words]

def extract_prediction_days(query: str, default_days: int = 30) -> int:
    match = re.search(r'(\d+)\s*day', query.lower())
    if match:
        days = int(match.group(1))
        return max(1, min(days, 365))
    else:
        return default_days

def should_predict(query):
    keywords = ["predict", "forecast", "future", "next month", "price after", "expected price", "real-time", "live"]
    return any(word in query.lower() for word in keywords)

def should_visualize(query):
    viz_keywords = [
        "trend", "chart", "graph", "visual", "visualization", "plot", "performance", 
        "price", "compare", "comparison", "history", "historical", "show me", 
        "display", "movement", "volatility", "correlation", "growth"
    ]
    return any(word in query.lower() for word in viz_keywords)

def is_real_time_query(query):
    keywords = ["real-time", "live", "current", "now", "today"]
    return any(word in query.lower() for word in keywords)

def extract_period_from_query(query):
    query = query.lower()
    if any(term in query for term in ["1 day", "today", "24 hour"]):
        return "1d"
    elif any(term in query for term in ["5 day", "week", "weekly"]):
        return "5d"
    elif any(term in query for term in ["1 month", "monthly", "30 day"]):
        return "1mo"
    elif any(term in query for term in ["3 month", "quarter", "quarterly"]):
        return "3mo"
    elif any(term in query for term in ["6 month", "half year"]):
        return "6mo"
    elif any(term in query for term in ["1 year", "annual", "yearly"]):
        return "1y"
    elif any(term in query for term in ["2 year", "2 years"]):
        return "2y"
    elif any(term in query for term in ["5 year", "long term"]):
        return "5y"
    elif any(term in query for term in ["10 year", "decade"]):
        return "10y"
    elif any(term in query for term in ["max", "all time", "inception"]):
        return "max"
    return "1mo"

def get_visualization_type(query):
    query = query.lower()
    if any(term in query for term in ["candlestick", "candle", "ohlc"]):
        return "candlestick"
    elif any(term in query for term in ["volume", "trading volume"]):
        return "volume"
    elif any(term in query for term in ["moving average", "ma", "ema", "sma"]):
        return "moving_average"
    elif any(term in query for term in ["compare", "comparison", "versus", "vs"]):
        return "comparison"
    elif any(term in query for term in ["correlation", "relate", "relationship"]):
        return "correlation"
    elif any(term in query for term in ["distribution", "histogram"]):
        return "distribution"
    elif any(term in query for term in ["volatility", "risk"]):
        return "volatility"
    return "line"

# LSTM Prediction Functions
def prepare_stock_data(data, sequence_length=60):
    data.fillna(method='ffill', inplace=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    X, Y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        Y.append(data_scaled[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    return X[:train_size], X[train_size:], Y[:train_size], Y[train_size:], scaler

def predict_stock_trend(ticker: str, prediction_days: int = 30, real_time: bool = False):
    try:
        # Determine data fetching strategy based on real-time flag
        if real_time:
            # Fetch recent data for real-time prediction (last 30 days with daily data)
            end = datetime.now()
            start = end - timedelta(days=180)  # Get 6 months of data for better training
            df = yf.download(ticker, start=start, end=end, interval="1d")
            if df.empty:
                return {"error": "No real-time data found for this ticker."}
        else:
            # Historical data for training
            start = '2020-01-01'  # Last ~5 years of data
            end = datetime.now().strftime('%Y-%m-%d')
            df = yf.download(ticker, start=start, end=end)
            if df.empty:
                return {"error": "No historical data found for this ticker."}

        # Check if we have enough data
        if len(df) < 60:  # Need at least 60 days for sequence
            return {"error": f"Not enough historical data for {ticker} to make predictions."}

        # Preprocessing - use the Close price for prediction
        df = df.dropna()
        
        # Create the scaler and scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        # Create sequences for LSTM training
        sequence_length = 60
        x_train, y_train = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            x_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # Reshape x_train to be 3D: [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Important: Reshape y_train to be 2D to match the model's output shape
        # y_train should be [samples, 1] to match the Dense(1) output
        y_train = y_train.reshape(-1, 1)
        
        # Build or load the LSTM model
        try:
            # Try to load existing model
            model = load_model('my_model.h5')
            print("Loaded existing model.")
        except:
            # Build new model if cannot load
            print("Building new LSTM model...")
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the model
            early_stopping = EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True
            )
            
            model.fit(
                x_train, y_train, 
                epochs=25, batch_size=32, 
                callbacks=[early_stopping], 
                verbose=0
            )
            
            # Save the model for future use
            model.save('keras_model.h5')
        
        # Prepare data for predictions
        # We use the last sequence_length days to predict future values
        last_sequence = scaled_data[-sequence_length:]
        future_input = last_sequence.reshape(1, sequence_length, 1)
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = []
        next_day = last_date
        while len(future_dates) < prediction_days:
            next_day += timedelta(days=1)
            if next_day.weekday() < 5:  # 0-4 are Monday to Friday
                future_dates.append(next_day)
        
        # Predict future prices
        predicted_prices = []
        
        for _ in range(prediction_days):
            # Make prediction for next day
            next_pred = model.predict(future_input, verbose=0)
            predicted_prices.append(next_pred[0, 0])
            
            # Update input sequence for next prediction
            # Reshape to match the expected input shape
            next_pred_reshaped = next_pred[0, 0].reshape(1, 1, 1)
            future_input = np.append(future_input[:, 1:, :], next_pred_reshaped, axis=1)
        
        # Transform predictions back to original scale
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        # Create prediction dataframe with actual dates
        predicted_df = pd.DataFrame({
            'Date': future_dates[:len(predicted_prices)],  # Ensure we don't exceed the length
            'Predicted_Price': predicted_prices.flatten()
        })
        
        # Get the last actual price for reference
        last_price = float(df['Close'].iloc[-1])

        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the predicted values
        ax.plot(predicted_df['Date'], predicted_df['Predicted_Price'], 
                'r--', label=f'Predicted {ticker} Price')
        
        # Add a marker showing the last actual price
        ax.axvline(x=df.index[-1], color='blue', linestyle='--', alpha=0.5)
        ax.annotate('Prediction Start', 
               (df.index[-1], min(predicted_df['Predicted_Price'])),
               xytext=(5, 10), textcoords='offset points',
               fontsize=10)
        
        # Calculate predicted change
        first_pred = predicted_df['Predicted_Price'].iloc[0]
        last_pred = predicted_df['Predicted_Price'].iloc[-1]
        change_percent = ((last_pred - first_pred) / first_pred) * 100
        
        # Add title and labels with prediction info
        title = f"{ticker} Price Prediction - Next {prediction_days} Days (Future Only)"
        if real_time:
            title = f"{ticker} Real-Time Price Prediction - Next {prediction_days} Days (Future Only)"
            
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        
        # Annotate the prediction trend
        color = 'green' if change_percent >= 0 else 'red'
        ax.annotate(f'Predicted Change: {change_percent:.2f}%', 
                   xy=(0.02, 0.05), xycoords='axes fraction',
                   fontsize=12, color=color)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        
        return {
            "plot": buf,
            "predicted_df": predicted_df,
            "last_actual_price": last_price,
            "predicted_change_percent": change_percent
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# Visualization functions
def plot_price_trend(symbol, period="1mo", query=""):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return None, None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data["Close"], label=f"{symbol} Close Price", linewidth=2)
        
        percent_change = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
        color = 'green' if percent_change >= 0 else 'red'
        
        ax.set_title(f"{symbol} Price Trend - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.annotate(f"{percent_change:.2f}%", xy=(data.index[-1], data["Close"].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points", color=color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        summary = pd.DataFrame({
            "Start": [f"${data['Close'].iloc[0]:.2f}"],
            "End": [f"${data['Close'].iloc[-1]:.2f}"],
            "Change": [f"{percent_change:.2f}%"]
        })
        
        return fig, summary
    except Exception as e:
        st.error(f"Error generating price trend: {e}")
        return None, None

def plot_candlestick(symbol, period="1mo", query=""):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        width = 0.6
        width2 = 0.05
        up = data[data.Close >= data.Open]
        down = data[data.Close < data.Open]
        
        ax.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color='green')
        ax.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color='green')
        ax.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color='green')
        ax.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color='red')
        ax.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color='red')
        ax.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color='red')
        
        ax.set_title(f"{symbol} Candlestick Chart - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating candlestick chart: {e}")
        return None

def plot_volume(symbol, period="1mo", query=""):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(data.index, data["Close"], label=f"{symbol} Close Price", color='blue')
        ax1.set_title(f"{symbol} Price and Volume - {period.upper()}", fontsize=16)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.bar(data.index, data["Volume"], color='purple', alpha=0.7)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating volume chart: {e}")
        return None

def plot_moving_average(symbol, period="1mo", query=""):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return None
            
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data.index, data["Close"], label=f"{symbol} Price", color='blue')
        ax.plot(data.index, data["MA10"], label="10-Day MA", color='red')
        ax.plot(data.index, data["MA20"], label="20-Day MA", color='green')
        ax.plot(data.index, data["MA50"], label="50-Day MA", color='purple')
        
        ax.set_title(f"{symbol} Price with Moving Averages - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating moving average chart: {e}")
        return None
    
class ChartData:
    def __init__(self, query, fig, description, chart_type, ticker, summary=None):
        self.query = query
        self.fig = fig
        self.description = description
        self.chart_type = chart_type
        self.ticker = ticker
        self.summary = summary

def plot_comparison(symbols, period="1mo", query=""):
    if not symbols or len(symbols) < 1:
        st.warning("No ticker symbols provided for comparison.")
        return None, None
        
    try:
        data_frames = {}
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty:
                data_frames[symbol] = df
        
        if not data_frames:
            st.warning("No data available for the specified symbols.")
            return None, None
        
        comparison_df = pd.DataFrame()
        for symbol, df in data_frames.items():
            if not df.empty:
                comparison_df[symbol] = (df['Close'] / df['Close'].iloc[0]) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_df.columns)))
        for i, symbol in enumerate(comparison_df.columns):
            ax.plot(comparison_df.index, comparison_df[symbol], label=symbol, linewidth=2, color=colors[i])
        
        ax.set_title(f"Performance Comparison (Normalized to 100) - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Price (Base=100)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        performance_data = {}
        for symbol in comparison_df.columns:
            percent_change = comparison_df[symbol].iloc[-1] - 100
            performance_data[symbol] = [f"{percent_change:.2f}%"]
        
        summary = pd.DataFrame(performance_data, index=["% Change"])
        
        return fig, summary
    except Exception as e:
        st.error(f"Error generating comparison chart: {e}")
        return None, None

def plot_correlation(symbols, period="1mo", query=""):
    if not symbols or len(symbols) < 2:
        st.warning("At least two ticker symbols are required for correlation analysis.")
        return None
        
    try:
        data_frames = {}
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty:
                data_frames[symbol] = df['Close'].pct_change().dropna()
        
        if len(data_frames) < 2:
            st.warning("Not enough data available for correlation analysis.")
            return None
        
        returns_df = pd.DataFrame(data_frames)
        corr_matrix = returns_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation Matrix of Daily Returns - {period.upper()}", fontsize=16)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating correlation matrix: {e}")
        return None

def plot_volatility(symbol, period="1mo", window=20, query=""):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return None
            
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily Return'].rolling(window=window).std() * np.sqrt(252) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(data.index, data["Close"], label=f"{symbol} Close Price", color='blue')
        ax1.set_title(f"{symbol} Price and {window}-Day Volatility - {period.upper()}", fontsize=16)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(data.index, data["Volatility"], color='red', label=f"{window}-Day Volatility")
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Annualized Volatility (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating volatility chart: {e}")
        return None

# Query submission logic
if st.button("Submit"):
    user_msg = HumanMessage(content=query)
    
    msg_index = len(st.session_state.history)
    
    st.session_state.history.append(user_msg)
    
    input_messages = [{"role": "user", "content": msg.content} for msg in st.session_state.history if isinstance(msg, HumanMessage)]
    
    output = app.invoke({
        "messages": input_messages,
        "language": language,
        "industrial_question": st.session_state.original_question
    })
    
    bot_reply = output["messages"][0].content
    
    tickers = extract_tickers(query)
    
    st.session_state.message_charts[msg_index] = []
    
    # Update this section in your main code where predictions are displayed
    if should_predict(query) and tickers:
        prediction_days = extract_prediction_days(query, default_days=30)
        is_real_time = is_real_time_query(query)

        for ticker in tickers:
            with st.spinner(f"Generating {'real-time ' if is_real_time else ''}{prediction_days}-day price predictions for {ticker}..."):
                prediction_output = predict_stock_trend(ticker, prediction_days=prediction_days, real_time=is_real_time)

                if "error" not in prediction_output:
                    prediction_table = prediction_output['predicted_df']
                    plot_buf = prediction_output['plot']
                    last_price = prediction_output['last_actual_price']
                    change_percent = prediction_output['predicted_change_percent']

                    if not prediction_table.empty:
                        # Prediction plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot the predicted values with actual dates
                        ax.plot(prediction_table['Date'], prediction_table['Predicted_Price'], 
                                label='Predicted Price', color='blue')
                        
                        ax.set_title(f"{ticker} {'Real-Time ' if is_real_time else ''}{prediction_days}-Day Price Prediction", 
                                    fontsize=16)
                        ax.set_xlabel("Date", fontsize=12)
                        ax.set_ylabel("Price ($)", fontsize=12)
                        
                        # Color the prediction trend
                        color = 'green' if change_percent >= 0 else 'red'
                        ax.annotate(f'Starting from ${last_price:.2f} on {prediction_table["Date"].iloc[0].strftime("%Y-%m-%d")}\n'
                                f'Predicted Change: {change_percent:.2f}%', 
                                xy=(0.02, 0.05), xycoords='axes fraction',
                                fontsize=12, color=color)
                        
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Rotate date labels for better readability
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                        chart_data = ChartData(query, fig, None, "prediction", ticker)
                        st.session_state.message_charts[msg_index].append(chart_data)

                        # Format the date column for display
                        formatted_df = prediction_table.copy()
                        formatted_df['Date'] = formatted_df['Date'].dt.strftime('%Y-%m-%d')
                        formatted_df.rename(columns={'Predicted_Price': 'Predicted Price ($)'}, inplace=True)
                        
                        # Display predicted prices in a table after the chart
                        st.subheader(f'Predicted Prices for {ticker} - Next {prediction_days} Days')
                        st.dataframe(formatted_df)
                        
                        # Add a summary of the prediction trend
                        trend_direction = "Upward" if change_percent >= 0 else "Downward"
                        st.info(f"The prediction shows a {trend_direction} trend with a {abs(change_percent):.2f}% change over the next {prediction_days} days.")
                            
                else:
                    st.error(prediction_output["error"])


    elif should_visualize(query) and tickers:
        period = extract_period_from_query(query)
        viz_type = get_visualization_type(query)
        
        if viz_type == "line":
            for ticker in tickers:
                fig, summary = plot_price_trend(ticker, period, query)
                if fig:
                    chart_data = ChartData(query, fig, summary, "price_trend", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "candlestick":
            for ticker in tickers:
                fig = plot_candlestick(ticker, period, query)
                if fig:
                    chart_data = ChartData(query, fig, None, "candlestick", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "volume":
            for ticker in tickers:
                fig = plot_volume(ticker, period, query)
                if fig:
                    chart_data = ChartData(query, fig, None, "volume", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "moving_average":
            for ticker in tickers:
                fig = plot_moving_average(ticker, period, query)
                if fig:
                    chart_data = ChartData(query, fig, None, "moving_average", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "comparison" and len(tickers) > 1:
            fig, summary = plot_comparison(tickers, period, query)
            if fig:
                chart_data = ChartData(query, fig, summary, "comparison", ",".join(tickers))
                st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "correlation" and len(tickers) > 1:
            fig = plot_correlation(tickers, period, query)
            if fig:
                chart_data = ChartData(query, fig, None, "correlation", ",".join(tickers))
                st.session_state.message_charts[msg_index].append(chart_data)
        elif viz_type == "volatility":
            for ticker in tickers:
                fig = plot_volatility(ticker, period, 20, query)
                if fig:
                    chart_data = ChartData(query, fig, None, "volatility", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="2d")
            if not data.empty:
                current_price = data["Close"].iloc[-1]
                previous_price = data["Close"].iloc[-2]
                change_value = current_price - previous_price
                change_percent = (change_value / previous_price) * 100
                
                stock_info = f"📈 {ticker}: ${current_price:.2f} ({change_percent:.2f}%)"
                bot_reply = stock_info + "\n\n" + bot_reply
        except:
            pass
    
    st.session_state.history.append(AIMessage(content=bot_reply))

# Display conversation with charts inline
def display_conversation():
    turns = []
    current_turn = {"user": None, "ai": None, "charts": []}
    
    for i, msg in enumerate(st.session_state.history):
        if isinstance(msg, HumanMessage):
            if current_turn["user"] is not None and current_turn["ai"] is not None:
                turns.append(current_turn.copy())
                current_turn = {"user": None, "ai": None, "charts": []}
            
            current_turn["user"] = {"index": i, "content": msg.content}
            
        elif isinstance(msg, AIMessage):
            current_turn["ai"] = {"index": i, "content": msg.content}
            
            if current_turn["user"] is not None:
                user_index = current_turn["user"]["index"]
                if user_index in st.session_state.message_charts:
                    current_turn["charts"] = st.session_state.message_charts[user_index]
    
    if current_turn["user"] is not None and current_turn["ai"] is not None:
        turns.append(current_turn)
    
    for turn in turns:
        st.markdown(f"👤 You: {turn['user']['content']}")
        st.markdown(f"🤖 AI: {turn['ai']['content']}")
        
        if turn["charts"]:
            with st.container():
                for chart in turn["charts"]:
                    st.subheader(f"{chart.chart_type.replace('_', ' ').title()} - {chart.ticker}")
                    st.pyplot(chart.fig)
                    if chart.summary is not None:
                        st.table(chart.summary)

display_conversation()

if st.sidebar.button("Clear All Charts"):
    st.session_state.message_charts = {}
    st.rerun()

st.header("Financial Data and Stock Screener")

symbol = st.sidebar.text_input("Ticker:", value="TSLA")

financial_data = st.sidebar.selectbox(
    "Select Financial Data Type", 
    options=(
        'income-statement', 'balance-sheet-statement', 'cash-flow-statement',
        'income-statement-growth', 'balance-sheet-statement-growth', 
        'cash-flow-statement-growth', 'ratios-ttm', 'ratios', 'financial-growth',
        'quote', 'rating', 'enterprise-values', 'key-metrics-ttm', 'key-metrics',
        'historical-rating', 'discounted-cash-flow', 'historical-discounted-cash-flow-statement',
        'historical-price-full', 'Historical Price smaller intervals'
    )
)

if financial_data == 'Historical Price smaller intervals':
    interval = st.sidebar.selectbox('Interval', options=('1min','5min', '15min', '30min','1hour', '4hour'))
    financial_data = 'historical-chart/' + interval

transpose = st.sidebar.selectbox('Transpose Data', options=('Yes', 'No'))

def fetch_stock_data(symbol, financial_data):
    url = f'{BASE_URL}/{API_VERSION}/{financial_data}/{symbol}?apikey={API_KEY}'
    try:
        response = requests.get(url)
        st.sidebar.write(f"Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if not data:
                st.error(f"No data returned for {symbol}")
            else:
                df = pd.DataFrame(data)
                return df
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            st.write(f"Response: {response.text}")
    except Exception as e:
        st.error(f"Error making API request: {str(e)}")
    return None

df = fetch_stock_data(symbol, financial_data)

if df is not None:
    if transpose == 'Yes':
        df = df.T
    st.write(df)

st.header('Indian Stock Dashboard')

# --- User Inputs ---
ticker = st.sidebar.text_input('Symbol Code (e.g., INFY, TSLA)', 'INFY')
exchange = st.sidebar.text_input('Exchange (e.g., NSE, NASDAQ)', 'NSE')

# --- Build Google Finance URL ---
url = f'https://www.google.com/finance/quote/{ticker}:{exchange}'

# --- Scrape Basic Info from Google Finance ---
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

def extract_text(class_name):
    tag = soup.find(class_=class_name)
    return tag.text if tag else 'N/A'

try:
    price = float(extract_text('YMlKec fxKbKc')[1:].replace(',', ''))
except:
    price = None

try:
    prev_close = float(extract_text('P6K39c')[1:].replace(',', ''))
except:
    prev_close = None

percent_change = ((price - prev_close) / prev_close * 100) if price and prev_close else None

info = {
    'Price': price if price else 'N/A',
    'Previous Close': prev_close if prev_close else 'N/A',
    '% Change': f"{percent_change:.2f}%" if percent_change else 'N/A',
    'News': extract_text('Yfwt5'),
    'About': extract_text('bLLb2d')
}

df_info = pd.DataFrame(info, index=['Value']).T
st.write(df_info)

# --- yfinance Symbol ---
# For NSE tickers, yfinance requires .NS suffix
yf_ticker = f"{ticker}.NS" if exchange.upper() == "NSE" else ticker

try:
    data = yf.download(yf_ticker, period='1mo', interval='1d')
    st.subheader('📊 Price Trend - Last 30 Days')
    st.line_chart(data['Close'])
except:
    st.error("Could not fetch price data for chart. Check the symbol and exchange.")