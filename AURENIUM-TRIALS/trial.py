# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from typing import TypedDict, Sequence

# --- CONFIGURATION ---
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

# OpenAI / LangSmith Keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])

class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str
    original_question: str

system_prompt = """
You are a financial assistant. Provide advice based on financial data.
Predict stock prices if asked. Generate visualizations if requested.
Keep answers professional and brief.
"""

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

# --- HELPER FUNCTIONS ---

# Extract Tickers
def extract_tickers(query):
    tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', query.upper())
    all_tickers = list(set(tickers + dollar_tickers))
    common_words = ["I", "A", "FOR", "IN", "ON", "AT", "BY", "AND", "OR", "THE", "TO"]
    return [ticker for ticker in all_tickers if ticker not in common_words]

# Should Predict
def should_predict(query):
    keywords = ["predict", "forecast", "future", "next month", "price after", "expected price"]
    return any(word in query.lower() for word in keywords)

# Should Visualize
def should_visualize(query):
    keywords = [
        "trend", "chart", "graph", "visual", "visualization", "plot",
        "performance", "price", "compare", "comparison", "history",
        "historical", "movement", "volatility", "correlation", "growth"
    ]
    return any(word in query.lower() for word in keywords)

# Extract Period
def extract_period(query):
    query = query.lower()
    if "1 day" in query or "today" in query: return "1d"
    elif "5 day" in query or "week" in query: return "5d"
    elif "1 month" in query: return "1mo"
    elif "3 month" in query or "quarter" in query: return "3mo"
    elif "6 month" in query: return "6mo"
    elif "1 year" in query: return "1y"
    elif "2 year" in query: return "2y"
    elif "5 year" in query: return "5y"
    elif "10 year" in query: return "10y"
    elif "max" in query or "all time" in query: return "max"
    else: return "1mo"

# Financial Data Fetch
def fetch_financial_data(symbol, financial_data):
    url = f'{BASE_URL}/{API_VERSION}/{financial_data}/{symbol}?apikey={API_KEY}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                return pd.DataFrame(data)
            else:
                st.warning(f"No data for {symbol}")
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

# Prepare Data
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

# Create LSTM
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

# Predict Future
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

# Plotting
def plot_price_trend(symbol, period="1mo"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, data['Close'])
    ax.set_title(f"{symbol} Price Trend ({period})")
    ax.set_ylabel('Price ($)')
    ax.grid()
    return fig

# --- STREAMLIT APP ---

st.set_page_config(page_title="Financial Assistant", layout="wide")
st.title("📈 Financial Chatbot and Screener")

# Sidebar
symbol = st.sidebar.text_input("Ticker Symbol:", value="AAPL")
financial_data = st.sidebar.selectbox(
    "Select Financial Data Type",
    options=[ 'income-statement', 'balance-sheet-statement', 'cash-flow-statement',
              'income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth',
              'ratios-ttm', 'ratios', 'financial-growth', 'quote', 'rating', 'enterprise-values',
              'key-metrics-ttm', 'key-metrics', 'historical-rating', 'discounted-cash-flow',
              'historical-discounted-cash-flow-statement']
)
transpose = st.sidebar.selectbox('Transpose Data?', options=('Yes', 'No'))

# Chatbot
if "history" not in st.session_state:
    st.session_state.history = []

language = st.selectbox("Language:", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask about stock...")

if st.button("Submit"):
    user_msg = HumanMessage(content=query)
    st.session_state.history.append(user_msg)

    tickers = extract_tickers(query)
    predict = should_predict(query)
    visualize = should_visualize(query)
    period = extract_period(query)

    if predict and tickers:
        for ticker in tickers:
            dates, prices = predict_future_prices(ticker)
            if dates is not None:
                st.success(f"📈 Predicted 30 days for {ticker}")
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(dates, prices)
                ax.set_title(f"{ticker} Future Price")
                st.pyplot(fig)
                pred_df = pd.DataFrame({"Date": dates, "Predicted Price": prices})
                st.dataframe(pred_df)

    elif visualize and tickers:
        for ticker in tickers:
            fig = plot_price_trend(ticker, period)
            if fig:
                st.pyplot(fig)

    else:
        input_messages = [{"role": "user", "content": msg.content} for msg in st.session_state.history]
        output = app.invoke({
            "messages": input_messages,
            "language": language,
            "original_question": None
        })
        bot_reply = output["messages"][0].content
        st.session_state.history.append(AIMessage(content=bot_reply))

# Show Conversation
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 **AI:** {msg.content}")

# Financial Table
st.header("📊 Financial Data")
df = fetch_financial_data(symbol, financial_data)
if df is not None:
    if transpose == 'Yes':
        df = df.T
    st.dataframe(df, use_container_width=True)
