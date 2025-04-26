#Provides stock prices and also provide specific stocks to invest in
import yfinance as yf
import os
import streamlit as st
import pandas as pd
import requests
import re
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
    response_format: str

# Enhanced system prompt with specific instructions
system_prompt = """
You are a financial risk assessment assistant. You provide investment advice based on financial information provided by the user.

IMPORTANT INSTRUCTIONS:
1. If the user explicitly requests answers in a specific format (one word, two words, short answer, etc.), strictly follow that format.
2. When comparing stocks, provide clear comparative analysis on key metrics like price, performance, volatility, and fundamentals.
3. For specific stock inquiries, provide concise but comprehensive analysis.
4. Adapt your response length precisely to what the user requests.
5. When users ask about stock prices, answer factually based on the real-time data provided to you.
6. If the user's language is not English, respond in their preferred language.

Your goal is to be helpful, accurate, and to match the exact level of detail and format requested by the user.
"""

# Set up LangChain Prompt with enhanced system message
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Function to extract formatting instructions from the query
def extract_formatting_instructions(query):
    # Patterns to recognize common formatting requests
    one_word_pattern = re.compile(r'(answer|respond|reply|tell me).{0,20}(one|1|single)\s*word', re.IGNORECASE)
    two_word_pattern = re.compile(r'(answer|respond|reply|tell me).{0,20}(two|2)\s*words', re.IGNORECASE)
    brief_pattern = re.compile(r'(answer|respond|reply|tell me).{0,20}(brief|short|concise|quickly)', re.IGNORECASE)
    
    format_instruction = "standard"
    
    if one_word_pattern.search(query):
        format_instruction = "one_word"
    elif two_word_pattern.search(query):
        format_instruction = "two_word"
    elif brief_pattern.search(query):
        format_instruction = "brief"
    elif "compare" in query.lower() and any(ticker in query.upper() for ticker in re.findall(r'\b[A-Z]{1,5}\b', query)):
        format_instruction = "comparison"
    
    return format_instruction

# Function to detect stock symbols in query
def detect_stock_symbols(query):
    # Look for capitalized words that could be stock tickers
    potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
    return potential_tickers

# Enhanced function to call LangChain model
def call_model(state: State):
    # Add formatting instructions to the system prompt
    response_format = state.get("response_format", "standard")
    
    prompt = prompt_template.invoke({
        "messages": state["messages"],
        "language": state.get("language", "English")
    })
    
    # Add additional context about formatting if needed
    if response_format != "standard":
        formatting_context = HumanMessage(content=f"[System Note: The user has requested a {response_format} response format. Please adjust your answer accordingly.]")
        prompt.messages.append(formatting_context)
    
    response = model.invoke(prompt)
    return {"messages": [response]}

# Initialize the workflow for LangChain
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

memory = MemorySaver()
app = workflow.compile()

# Enhanced stock price fetching to support comparison
def fetch_stock_data_for_comparison(symbols):
    results = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="5d")  # Fetch 5 days for better trend analysis
            
            if data.empty:
                results[symbol] = {"price": "N/A", "change_percent": "N/A", "error": True}
                continue
                
            current_price = data["Close"].iloc[-1]
            previous_price = data["Close"].iloc[-2]
            week_ago_price = data["Close"].iloc[0] if len(data) >= 5 else previous_price
            
            daily_change_value = current_price - previous_price
            daily_change_percent = (daily_change_value / previous_price) * 100
            
            weekly_change_value = current_price - week_ago_price
            weekly_change_percent = (weekly_change_value / week_ago_price) * 100
            
            # Get additional info if available
            info = stock.info
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            volume = data["Volume"].iloc[-1] if not data["Volume"].empty else 'N/A'
            
            results[symbol] = {
                "price": f"{current_price:.2f}",
                "daily_change_value": f"{daily_change_value:.2f}",
                "daily_change_percent": f"{daily_change_percent:.2f}%",
                "weekly_change_percent": f"{weekly_change_percent:.2f}%",
                "volume": volume,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "error": False
            }
        except Exception as e:
            results[symbol] = {"price": "N/A", "error": True, "error_message": str(e)}
    
    return results

# Fetch real-time stock data using yfinance (original function preserved)
def fetch_realtime_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="2d")  # Fetch the last 2 days' data to calculate change
        
        if data.empty:  # If no data is returned (invalid ticker)
            return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}
        
        current_price = data["Close"].iloc[-1]  # Get the most recent closing price
        previous_price = data["Close"].iloc[-2]  # Get the previous closing price
        
        change_value = current_price - previous_price  # Calculate the absolute change
        change_percent = (change_value / previous_price) * 100  # Calculate the percentage change
        
        # Format the output to display value and percentage change
        return {
            "price": f"{current_price:.2f}",
            "change_value": f"{change_value:.2f}",
            "change_percent": f"{change_percent:.2f}%",
            "error": False  # No error, data is valid
        }
    except Exception:
        return {"price": "N/A", "change_value": "N/A", "change_percent": "N/A", "error": True}

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

    # Extract formatting instructions
    response_format = extract_formatting_instructions(query)
    
    # Detect stock symbols for potential comparison
    stock_symbols = detect_stock_symbols(query)
    
    # Prepare additional context for the model
    stock_data_context = ""
    
    # Handle stock comparison if multiple symbols detected and comparison terms in query
    if len(stock_symbols) >= 2 and ("compare" in query.lower() or "vs" in query.lower() or "versus" in query.lower()):
        comparison_data = fetch_stock_data_for_comparison(stock_symbols)
        
        # Format comparison data for the model
        stock_data_context = "Stock comparison data:\n"
        for symbol, data in comparison_data.items():
            if not data["error"]:
                stock_data_context += f"- {symbol}: Price=${data['price']}, Daily Change: {data['daily_change_percent']}, Weekly Change: {data['weekly_change_percent']}"
                if data['market_cap'] != 'N/A':
                    stock_data_context += f", Market Cap: {data['market_cap']}"
                if data['pe_ratio'] != 'N/A':
                    stock_data_context += f", P/E Ratio: {data['pe_ratio']}"
                stock_data_context += "\n"
            else:
                stock_data_context += f"- {symbol}: Data unavailable\n"
    
    # Prepare input messages
    input_messages = [{"role": "system", "content": "You are a financial risk assessment assistant."}]
    input_messages.extend([{"role": "user", "content": msg.content} for msg in st.session_state.history])
    
    # Add stock data context if available
    if stock_data_context:
        input_messages.append({"role": "system", "content": stock_data_context})

    # Invoke the model with all context
    output = app.invoke({
        "messages": input_messages,
        "language": language,
        "original_question": st.session_state.original_question,
        "response_format": response_format
    })

    bot_reply = output["messages"][0].content

    # Dynamically detect tickers in the user query and fetch their real-time price
    # Only add stock price info if not already doing a comparison
    if not (len(stock_symbols) >= 2 and ("compare" in query.lower() or "vs" in query.lower() or "versus" in query.lower())):
        for word in stock_symbols:
            live = fetch_realtime_stock_price(word)  # Convert to uppercase for tickers
            if not live["error"]:  # Valid stock data
                bot_reply = (
                    f"📈 The current stock price of {word} is ${live['price']}, "
                    f"with a change of {live['change_percent']}.\n\n" + bot_reply
                )
            else:  # Invalid stock ticker
                bot_reply = f"⚠ Stock data not available for {word}.\n\n" + bot_reply

    st.session_state.history.append(AIMessage(content=bot_reply))

# Display the conversation history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"👤 You: {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"🤖 AI: {msg.content}")

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