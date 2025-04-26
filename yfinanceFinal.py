import yfinance as yf
import requests
import os
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# API Configuration
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

# LangSmith and Cohere setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

os.environ["GROK_API_KEY"] = "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"  # Replace with your Grok API key
os.environ["GROK_API_BASE"] = "https://api.groq.com/openai/v1"  # Replace with the actual Grok API endpoint

# Initialize Grok model (change the model name according to Grok's offerings)
model = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Or a different Grok model name
    openai_api_key=os.environ["GROK_API_KEY"],  # Using Grok API key here
    openai_api_base=os.environ["GROK_API_BASE"],  # Grok API base URL
)
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
For price predictions, acknowledge that you'll generate LSTM-based forecasts.
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
    st.session_state.message_charts = {}  # Changed from charts to message_charts with message index as key

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask about stocks or request visualizations:")

# Helper functions
def extract_tickers(query):
    standard_tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', query)
    all_tickers = list(set(standard_tickers + dollar_tickers))
    common_words = ["I", "A", "FOR", "IN", "ON", "AT", "BY", "AND", "OR", "THE", "TO"]
    return [ticker for ticker in all_tickers if ticker not in common_words]

def should_predict(query):
    keywords = ["predict", "forecast", "future", "next month", "price after", "expected price"]
    return any(word in query.lower() for word in keywords)

def should_visualize(query):
    viz_keywords = [
        "trend", "chart", "graph", "visual", "visualization", "plot", "performance", 
        "price", "compare", "comparison", "history", "historical", "show me", 
        "display", "movement", "volatility", "correlation", "growth"
    ]
    return any(word in query.lower() for word in viz_keywords)

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
    return "1mo"  # Default

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
    return "line"  # Default

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

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(ticker, days=30, sequence_length=60):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")
    if data.empty:
        return None, None
    X_train, X_test, Y_train, Y_test, scaler = prepare_stock_data(data, sequence_length)
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stop])

    closing_prices = data['Close'].values
    scaled_data = scaler.transform(closing_prices.reshape(-1,1))
    current_batch = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    predictions = []
    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)
        predictions.append(scaler.inverse_transform(pred)[0,0])
        current_batch = np.append(current_batch[:,1:,:], pred.reshape(1,1,1), axis=1)

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    return future_dates, predictions

# Create a ChartData class to store chart information
class ChartData:
    def __init__(self, query, fig, summary=None, chart_type=None, ticker=None):
        self.query = query
        self.fig = fig
        self.summary = summary
        self.chart_type = chart_type
        self.ticker = ticker
        self.timestamp = datetime.now()

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
        
        # Calculate percent change
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
        
        # Summary stats
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
        
        # Plot candles
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
        
        # Set subplot height: 3 parts for price chart, 1 part for volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data["Close"], label=f"{symbol} Close Price", color='blue')
        ax1.set_title(f"{symbol} Price and Volume - {period.upper()}", fontsize=16)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Volume chart
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
            
        # Calculate moving averages
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

def plot_comparison(symbols, period="1mo", query=""):
    if not symbols or len(symbols) < 1:
        st.warning("No ticker symbols provided for comparison.")
        return None, None
        
    try:
        # Get data for all symbols
        data_frames = {}
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty:
                data_frames[symbol] = df
        
        if not data_frames:
            st.warning("No data available for the specified symbols.")
            return None, None
        
        # Create normalized prices dataframe
        comparison_df = pd.DataFrame()
        for symbol, df in data_frames.items():
            if not df.empty:
                comparison_df[symbol] = (df['Close'] / df['Close'].iloc[0]) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each stock
        colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_df.columns)))
        for i, symbol in enumerate(comparison_df.columns):
            ax.plot(comparison_df.index, comparison_df[symbol], label=symbol, linewidth=2, color=colors[i])
        
        ax.set_title(f"Performance Comparison (Normalized to 100) - {period.upper()}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Price (Base=100)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        # Performance summary
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
        # Get returns data
        data_frames = {}
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if not df.empty:
                data_frames[symbol] = df['Close'].pct_change().dropna()
        
        if len(data_frames) < 2:
            st.warning("Not enough data available for correlation analysis.")
            return None
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame(data_frames)
        corr_matrix = returns_df.corr()
        
        # Create heatmap
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
            
        # Calculate volatility
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily Return'].rolling(window=window).std() * np.sqrt(252) * 100  # Annualized
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data["Close"], label=f"{symbol} Close Price", color='blue')
        ax1.set_title(f"{symbol} Price and {window}-Day Volatility - {period.upper()}", fontsize=16)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Volatility chart
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
    
    # Current message index for chart mapping
    msg_index = len(st.session_state.history)
    
    # Add user message to history
    st.session_state.history.append(user_msg)
    
    # Prepare messages for LangChain
    input_messages = [{"role": "user", "content": msg.content} for msg in st.session_state.history if isinstance(msg, HumanMessage)]
    
    # Get response from LLM
    output = app.invoke({
        "messages": input_messages,
        "language": language,
        "original_question": st.session_state.original_question
    })
    
    bot_reply = output["messages"][0].content
    
    # Extract ticker symbols and check if visualization is needed
    tickers = extract_tickers(query)
    
    # Initialize charts for this message
    st.session_state.message_charts[msg_index] = []
    
    # Handle prediction requests
    if should_predict(query) and tickers:
        for ticker in tickers:
            with st.spinner(f"Generating price predictions for {ticker}..."):
                dates, prices = predict_future_prices(ticker)
                if dates is not None:
                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.plot(dates, prices, label='Predicted Price', color='blue')
                    ax.set_title(f"{ticker} 30-Day Price Prediction", fontsize=16)
                    ax.set_xlabel("Date", fontsize=12)
                    ax.set_ylabel("Price ($)", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    
                    chart_data = ChartData(query, fig, None, "prediction", ticker)
                    st.session_state.message_charts[msg_index].append(chart_data)
                    
                    pred_df = pd.DataFrame({"Date": dates, "Predicted Price": prices})
                    bot_reply += f"\n\nPredicted prices for {ticker}:\n{pred_df.to_string(index=False)}"
    
    # Create visualizations based on query
    elif should_visualize(query) and tickers:
        period = extract_period_from_query(query)
        viz_type = get_visualization_type(query)
        
        # Generate appropriate visualization and store in session state
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
    
    # Add stock price data to reply if tickers found
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="2d")
            if not data.empty:
                current_price = data["Close"].iloc[-1]
                previous_price = data["Close"].iloc[-2]
                change_value = current_price - previous_price
                change_percent = (change_value / previous_price) * 100
                
                # Add stock info to response
                stock_info = f"📈 *{ticker}*: ${current_price:.2f} ({change_percent:.2f}%)"
                bot_reply = stock_info + "\n\n" + bot_reply
        except:
            pass
    
    st.session_state.history.append(AIMessage(content=bot_reply))

# Display conversation with charts inline
for i, msg in enumerate(st.session_state.history):
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 *You:* {msg.content}")
        
        # Display charts right after the query they're related to
        if i in st.session_state.message_charts and st.session_state.message_charts[i]:
            with st.container():
                charts = st.session_state.message_charts[i]
                for chart in charts:
                    st.subheader(f"{chart.chart_type.replace('_', ' ').title()} - {chart.ticker}")
                    st.pyplot(chart.fig)
                    if chart.summary is not None:
                        st.table(chart.summary)
    
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 *AI:* {msg.content}")

# Add option to clear all charts
if st.sidebar.button("Clear All Charts"):
    st.session_state.message_charts = {}
    st.rerun()

# Set up header for the application
st.header("Financial Data and Stock Screener")

# Sidebar for stock data selection
symbol = st.sidebar.text_input("Ticker:", value="TSLA")

# Financial data type selector
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

# Handle small interval data selection
if financial_data == 'Historical Price smaller intervals':
    interval = st.sidebar.selectbox('Interval', options=('1min','5min', '15min', '30min','1hour', '4hour'))
    financial_data = 'historical-chart/' + interval

transpose = st.sidebar.selectbox('Transpose Data', options=('Yes', 'No'))

# Fetch stock data
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

# Display fetched stock data
df = fetch_stock_data(symbol, financial_data)

if df is not None:
    if transpose == 'Yes':
        df = df.T
    st.write(df)

# Indian Stock Dashboard Section (Google Finance Only)
st.header('Indian Stock Dashboard')

# Inputs for Google Finance
ticker = st.sidebar.text_input('Symbol Code','INFY')
exchange = st.sidebar.text_input('Exchange','NSE')

# Fetch data from Google Finance
url = f'https://www.google.com/finance/quote/{ticker}:{exchange}'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract relevant data
price = float(soup.find(class_='YMlKec fxKbKc').text.strip()[1:].replace(",",""))
previous_close = float(soup.find(class_='P6K39c').text.strip()[1:].replace(",",""))
revenue = soup.find(class_='QXDnM').text
news = soup.find(class_='Yfwt5').text
about = soup.find(class_='bLLb2d').text

# Create DataFrame for Google Finance data
# Calculate percentage change
percent_change = ((price - previous_close) / previous_close) * 100

# Create DataFrame for Google Finance data
dict1 = {
    'Price': price,
    'Previous Price': previous_close,
    'Revenue': revenue,
    'News': news,
    'About': about,
    '% Change': f"{percent_change:.2f}%"  # Format to 2 decimal places
}

df_google_finance = pd.DataFrame(dict1, index=['Extracted Data']).T

# Display the data
st.write(df_google_finance)
