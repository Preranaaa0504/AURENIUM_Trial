import yfinance as yf
import os
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing import TypedDict, Sequence
import re

# API Configuration
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

# LangSmith and OpenAI setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize LangChain model
model = init_chat_model(
    model="openai/gpt-3.5-turbo",
    model_provider="openai"
)

# Define LangChain State
class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str
    original_question: str

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages([ 
    ("system", "You are a financial risk assessment assistant. You provide investment advice based on financial information provided by the user. Ask questions about their financial situation and give stock advice considering these details."),
    MessagesPlaceholder(variable_name="messages"),
])

def call_model(state: State):
    prompt = prompt_template.invoke({"messages": state["messages"], "language": state.get("language", "English")})
    response = model.invoke(prompt)
    return {"messages": [response]}

# LangGraph Workflow
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

memory = MemorySaver()
app = workflow.compile()

# Streamlit Setup
st.set_page_config(page_title="Financial Risk Assessment Chatbot", layout="centered")
st.title("Financial Risk Assessment Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

if "original_question" not in st.session_state:
    st.session_state.original_question = None

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask something about your financial plans:")

# Helper Functions
def fetch_realtime_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="2d")
        if data.empty:
            return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}
        current_price = data["Close"].iloc[-1]
        previous_price = data["Close"].iloc[-2]
        change_value = current_price - previous_price
        change_percent = (change_value / previous_price) * 100
        return {
            "price": f"{current_price:.2f}",
            "change_value": f"{change_value:.2f}",
            "change_percent": f"{change_percent:.2f}%",
            "error": False
        }
    except Exception:
        return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}

def should_plot(query):
    plot_keywords = ["trend", "chart", "graph", "visual", "performance", "price", "compare", "comparison", "history"]
    return any(word in query.lower() for word in plot_keywords)

def extract_period_from_query(query):
    query = query.lower()
    if "1 day" in query or "today" in query:
        return "1d"
    elif "5 day" in query or "this week" in query:
        return "5d"
    elif "1 month" in query or "last month" in query:
        return "1mo"
    elif "3 month" in query or "quarter" in query:
        return "3mo"
    elif "6 month" in query or "half year" in query:
        return "6mo"
    elif "1 year" in query or "last year" in query:
        return "1y"
    elif "2 year" in query:
        return "2y"
    elif "5 year" in query:
        return "5y"
    elif "10 year" in query:
        return "10y"
    elif "max" in query or "all time" in query:
        return "max"
    return "1mo"

def extract_tickers(query):
    return re.findall(r"\\b[A-Z]{2,5}\\b", query)

def plot_price_trend(symbol, period="1mo"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.warning(f"No historical data available for {symbol}.")
            return
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data["Close"], label=f"{symbol} Close", marker='o')
        plt.title(f"{symbol} Price Trend - Period: {period}")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating plot: {e}")

# Query submission logic
if st.button("Submit"):
    user_msg = HumanMessage(content=query)
    st.session_state.history.append(user_msg)
    input_messages = [{"role": "system", "content": "You are a financial risk assessment assistant."}]
    input_messages.extend([{"role": "user", "content": msg.content} for msg in st.session_state.history])

    output = app.invoke({
        "messages": input_messages,
        "language": language,
        "original_question": st.session_state.original_question
    })

    bot_reply = output["messages"][0].content

    tickers = extract_tickers(query)
    if should_plot(query):
        period = extract_period_from_query(query)
        for ticker in tickers:
            plot_price_trend(ticker, period)

    for ticker in tickers:
        live = fetch_realtime_stock_price(ticker)
        if not live["error"]:
            bot_reply = (
                f"📈 The current stock price of **{ticker}** is **${live['price']}**, "
                f"with a change of **{live['change_percent']}**.\n\n" + bot_reply
            )
        else:
            bot_reply = f"⚠️ Stock data not available for **{ticker}**.\n\n" + bot_reply

    st.session_state.history.append(AIMessage(content=bot_reply))

# Display conversation
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 **AI:** {msg.content}")

# Sidebar for Financial Screener
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

st.header("Financial Data and Stock Screener")
symbol = st.sidebar.text_input("Ticker:", value="TSLA")
financial_data = st.sidebar.selectbox("Select Financial Data Type", [
    'income-statement', 'balance-sheet-statement', 'cash-flow-statement',
    'income-statement-growth', 'balance-sheet-statement-growth', 
    'cash-flow-statement-growth', 'ratios-ttm', 'ratios', 'financial-growth',
    'quote', 'rating', 'enterprise-values', 'key-metrics-ttm', 'key-metrics',
    'historical-rating', 'discounted-cash-flow', 'historical-discounted-cash-flow-statement',
    'historical-price-full', 'Historical Price smaller intervals']
)
if financial_data == 'Historical Price smaller intervals':
    interval = st.sidebar.selectbox('Interval', ['1min', '5min', '15min', '30min', '1hour', '4hour'])
    financial_data = 'historical-chart/' + interval

transpose = st.sidebar.selectbox('Transpose Data', ['Yes', 'No'])
df = fetch_stock_data(symbol, financial_data)
if df is not None:
    if transpose == 'Yes':
        df = df.T
    st.write(df)
