#No stock price prediction in real time
from typing import TypedDict, Sequence
import os
import streamlit as st
import requests
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing import Annotated

# Financial Modeling Prep API
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

# LangSmith and DeepSeek-compatible API Keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize LangChain model
model = init_chat_model(
    model="openai/gpt-3.5-turbo",
    model_provider="openai"
)

#state schema for the chatbot
class State(TypedDict):
    messages: Sequence[BaseMessage]  
    language: str
    original_question: str  # Store the original stock-related question

# Prompt for financial assistant
prompt_template = ChatPromptTemplate.from_messages([ 
    (
        "system",
        "You are a financial risk assessment assistant. You provide investment advice based on financial information provided by the user. "
        "Ask questions about their financial situation (age, income, risk tolerance) and give stock advice considering these details."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Function to process the chatbot interaction
def call_model(state: State):
    prompt = prompt_template.invoke({
        "messages": state["messages"], 
        "language": state.get("language", "English")
    })
    response = model.invoke(prompt)
    return {"messages": [response]}

# Create prompt(using current message and lang) → Send prompt to model → Receive AI response

#  LangGraph workflow
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

# Compile app
memory = MemorySaver()
app = workflow.compile()

# --- Function to fetch stock data from Financial Modeling Prep ---
def fetch_stock_data(symbol, financial_data):
    url = f'{BASE_URL}/{API_VERSION}/{financial_data}/{symbol}?apikey={API_KEY}'
    
    try:
        response = requests.get(url)
        # Debug information
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

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Risk Assessment Chatbot", layout="centered")
st.title("Financial Risk Assessment Chatbot")

# Initialize or update history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Store the original query for reference
if "original_question" not in st.session_state:
    st.session_state.original_question = None

language = st.selectbox("Select Language", ["English", "Hindi", "Spanish"], index=0)

query = st.text_input("Ask something about your financial plans:")

if st.button("Submit"):
    # If the user asks the stock question, store the original question
    if "buy tesla stock" in query.lower() and st.session_state.original_question is None:
        st.session_state.original_question = query
    
    # Ensure 'role' and 'content' are provided for each message
    user_msg = HumanMessage(content=query)
    
    # Append user message to history
    st.session_state.history.append(user_msg)

    # Include all messages in history for context
    input_messages = [{"role": "system", "content": "You are a financial risk assessment assistant."}]
    
    # Add previous conversation messages to the input context
    input_messages.extend([{"role": "user", "content": msg.content} for msg in st.session_state.history])

    # Call the workflow with updated history and original question
    output = app.invoke({"messages": input_messages, "language": language, "original_question": st.session_state.original_question})

    # Add the AI response to the history
    st.session_state.history.append(AIMessage(content=output["messages"][0].content))

# Show chat history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 **You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 **AI:** {msg.content}")

# --- Stock Screener Section ---
st.header("📊 Financial Data and Stock Screener")

symbol = st.sidebar.text_input("Ticker:", value="AAPL")

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

# Fetch data based on the selected financial data type
df = fetch_stock_data(symbol, financial_data)

if df is not None:
    if transpose == 'Yes':
        df = df.T
    st.write(df)
