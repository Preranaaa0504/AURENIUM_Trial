import os
import streamlit as st
import pandas as pd
import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Sequence

# Alpha Vantage API for real-time prices
ALPHA_VANTAGE_API_KEY = 'MJY9U5NHJJ16ALMP'
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

# Financial Modeling Prep API for financial tables
FMP_API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'
FMP_BASE_URL = 'https://financialmodelingprep.com/api'
FMP_API_VERSION = 'v3'

# LangSmith & OpenAI Setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# LangChain Model
model = init_chat_model(model="openai/gpt-3.5-turbo", model_provider="openai")

# LangChain State
class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str
    original_question: str

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages([ 
    ("system", "You are a financial risk assessment assistant. Ask about financial status (age, income, risk tolerance), then provide investment advice."),
    MessagesPlaceholder(variable_name="messages"),
])

def call_model(state: State):
    prompt = prompt_template.invoke({
        "messages": state["messages"], 
        "language": state.get("language", "English")
    })
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")
app = workflow.compile()
memory = MemorySaver()

# Fetch real-time stock price from Alpha Vantage
def fetch_realtime_stock_price(symbol):
    try:
        url = f'{ALPHA_VANTAGE_BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (Daily)' not in data:
                return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}
            time_series = data['Time Series (Daily)']
            dates = list(time_series.keys())
            latest_data = time_series[dates[0]]
            previous_data = time_series[dates[1]]
            current_price = float(latest_data['4. close'])
            previous_price = float(previous_data['4. close'])
            change_value = current_price - previous_price
            change_percent = (change_value / previous_price) * 100
            return {
                "price": f"{current_price:.2f}",
                "change_value": f"{change_value:.2f}",
                "change_percent": f"{change_percent:.2f}%",
                "error": False
            }
    except Exception:
        pass
    return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}

# Fetch data from FMP
def fetch_fmp_data(symbol, endpoint):
    url = f"{FMP_BASE_URL}/{FMP_API_VERSION}/{endpoint}/{symbol}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Exception: {e}")
    return None

# Streamlit UI
st.set_page_config(page_title="Financial Risk Chatbot", layout="centered")
st.title("💸 Financial Risk Assessment Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)
query = st.text_input("Ask about your financial status or a stock:")

if st.button("Submit"):
    user_msg = HumanMessage(content=query)
    st.session_state.history.append(user_msg)
    input_messages = [{"role": "system", "content": "You are a financial risk assistant."}]
    input_messages.extend([{"role": "user", "content": msg.content} for msg in st.session_state.history])
    output = app.invoke({
        "messages": input_messages,
        "language": language,
        "original_question": query
    })
    bot_reply = output["messages"][0].content
    words = query.split()
    for word in words:
        if word.isupper():
            live = fetch_realtime_stock_price(word.strip())
            if not live["error"]:
                bot_reply = (
                    f"📈 The current stock price of **{word.upper()}** is **${live['price']}**, "
                    f"with a change of **{live['change_value']}** (**{live['change_percent']}**) today.\n\n" + bot_reply
                )
            else:
                bot_reply = f"⚠️ Stock data not available for **{word.upper()}**.\n\n" + bot_reply
    st.session_state.history.append(AIMessage(content=bot_reply))

for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 **AI:** {msg.content}")

# Financial Data Viewer (FMP)
st.header("📊 Financial Data Viewer")
symbol = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL")
data_type = st.sidebar.selectbox(
    "Select Financial Data Type (from FMP)", 
    options=[
        'income-statement', 'balance-sheet-statement', 'cash-flow-statement',
        'key-metrics', 'financial-growth', 'ratios', 'enterprise-values',
        'rating', 'quote', 'discounted-cash-flow', 'historical-price-full'
    ]
)
transpose = st.sidebar.checkbox("Transpose Table", value=False)

if symbol and data_type:
    df = fetch_fmp_data(symbol, data_type)
    if df is not None:
        if transpose:
            df = df.T
        st.subheader(f"📄 Data: {data_type.replace('-', ' ').title()} for {symbol.upper()}")
        st.write(df)
