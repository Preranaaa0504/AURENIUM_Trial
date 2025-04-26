import os
import pandas as pd
import streamlit as st
import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing import TypedDict, Sequence

# Define API Configuration
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'
ALPHA_VANTAGE_API_KEY = 'N81EYEWKOSM8RY8K'  # Replace with your Alpha Vantage API Key

# Set up LangSmith and DeepSeek-compatible API Keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize LangChain model
model = init_chat_model(
    model="openai/gpt-3.5-turbo",
    model_provider="openai"
)

# Define State for LangChain
class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str
    original_question: str

# Set up LangChain Prompt
prompt_template = ChatPromptTemplate.from_messages([ 
    (
        "system",
        "You are a financial risk assessment assistant. You provide investment advice based on financial information provided by the user. "
        "Ask questions about their financial situation (age, income, risk tolerance) and give stock advice considering these details."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Function to call LangChain model
def call_model(state: State):
    prompt = prompt_template.invoke({
        "messages": state["messages"], 
        "language": state.get("language", "English")
    })
    response = model.invoke(prompt)
    return {"messages": [response]}

# Initialize the workflow for LangChain
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

memory = MemorySaver()
app = workflow.compile()

# Fetch real-time stock price using Alpha Vantage API
def fetch_realtime_stock_price(symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "Time Series (5min)" in data:
            latest_time = max(data["Time Series (5min)"].keys())  # Get the most recent time
            latest_data = data["Time Series (5min)"][latest_time]
            current_price = latest_data["4. close"]
            change_value = float(current_price) - float(latest_data["1. open"])
            change_percent = (change_value / float(latest_data["1. open"])) * 100

            return {
                "price": current_price,
                "change_value": round(change_value, 2),
                "change_percent": round(change_percent, 2)
            }
        else:
            return {"error": "Data not available for this symbol."}
    except Exception as e:
        return {"error": str(e)}

# Streamlit app setup
st.set_page_config(page_title="Financial Risk Assessment Chatbot", layout="centered")
st.title("Financial Risk Assessment Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

if "original_question" not in st.session_state:
    st.session_state.original_question = None

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask something about your financial plans:")

# Submit button to process the query
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

    # Detect query and fetch stock data if it's related to stock prices
    if "price" in query.lower() and "stock" in query.lower():
        stock_symbol = query.split(" ")[-1].upper()  # Assuming the last word is the stock symbol
        stock_data = fetch_realtime_stock_price(stock_symbol)
        
        if "error" in stock_data:
            bot_reply = f"⚠️ Error fetching stock data: {stock_data['error']}"
        else:
            bot_reply = (
                f"📈 The current price of **{stock_symbol}** is **${stock_data['price']}**. "
                f"Change: **${stock_data['change_value']}** (Change: {stock_data['change_percent']}%)"
            )

    st.session_state.history.append(AIMessage(content=bot_reply))

# Display the conversation history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 **AI:** {msg.content}")

st.header("Financial Data and Stock Screener")

# Sidebar for additional stock data selection
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
