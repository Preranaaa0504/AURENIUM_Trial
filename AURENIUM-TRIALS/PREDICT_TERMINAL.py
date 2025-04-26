import os
import requests
import json
from typing import Sequence
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing_extensions import Annotated, TypedDict

# ✅ ENVIRONMENT VARIABLES
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
ALPHA_VANTAGE_API_KEY = "MJY9U5NHJJ16ALMP"

# ✅ CHAT MODEL INIT
model = ChatOpenAI(
    model="gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)

# ✅ STATE SCHEMA
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# ✅ PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a financial stock advisor assistant. Help the user make smart stock decisions in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ✅ TOKEN TRIMMER
def dummy_token_counter(messages):
    return sum(len(msg.content.split()) // 0.75 for msg in messages)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=dummy_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# ✅ INITIAL MODEL STEP WITH ERROR HANDLING
def call_model(state: State):
    try:
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({"messages": trimmed_messages, "language": state.get("language", "English")})
        response = model.invoke(prompt)
        if response is None:
            return {"messages": [AIMessage(content="Error: Received null response from model.")]}
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error calling model: {str(e)}")]}

# ✅ STOCK ANALYSIS STEP WITH ERROR HANDLING
def analyze_stock(state: State):
    user_input = state["messages"][-1].content
    stock_name = extract_stock_symbol(user_input)
    
    if not stock_name:
        return {"messages": [AIMessage(content="Please mention a valid stock ticker (e.g., AAPL, TSLA, GOOGL).")]}
    
    try:
        # Get stock data
        overview_response = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock_name}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        overview = overview_response.json()
        
        quote_response = requests.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_name}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        quote = quote_response.json().get("Global Quote", {})
        
        # Check if we got valid data
        if not overview or "Symbol" not in overview or not quote or "05. price" not in quote:
            return {"messages": [AIMessage(content=f"Could not fetch complete data for {stock_name}. Please check if it's a valid ticker.")]}
        
        # Parse data
        price = float(quote.get("05. price", "0"))
        change_percent = float(quote.get("10. change percent", "0").strip('%'))
        
        # Handle missing PE ratio
        try:
            pe_ratio = float(overview.get("PERatio", "0"))
        except ValueError:
            pe_ratio = 0
            
        market_cap = overview.get("MarketCapitalization", "N/A")
        
        # Simple rule-based decision
        if pe_ratio < 15 and change_percent > 0:
            recommendation = "Buy – The stock is undervalued and gaining momentum."
        elif pe_ratio > 30 or change_percent < -3:
            recommendation = "Sell – Overvalued or in decline."
        else:
            recommendation = "Hold – No clear indicator to buy or sell right now."
        
        response = (
            f"📈 **{stock_name.upper()} Stock Insight**\n"
            f"Price: ${price:.2f}\n"
            f"Change: {change_percent}%\n"
            f"P/E Ratio: {pe_ratio}\n"
            f"Market Cap: {market_cap}\n\n"
            f"📊 **Advice:** {recommendation}"
        )
        
        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error fetching stock data: {str(e)}. Please try again later.")]}

# ✅ SIMPLE SYMBOL EXTRACTOR
def extract_stock_symbol(text):
    words = text.upper().split()
    for word in words:
        if word.isalpha() and len(word) <= 5:
            return word
    return None

# ✅ TEST OPENROUTER CONNECTION
def test_openrouter_connection():
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"OpenRouter API test failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error testing OpenRouter connection: {str(e)}")
        return False

# ✅ WORKFLOW GRAPH
def create_workflow():
    workflow = StateGraph(state_schema=State)
    workflow.add_node("model", call_model)
    workflow.add_node("stock_analysis", analyze_stock)
    workflow.set_entry_point("model")
    workflow.add_edge("model", "stock_analysis")
    workflow.set_finish_point("stock_analysis")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ✅ EXAMPLE CALL
def run_stock_example():
    print("=== Real-Time Stock Advisor ===")
    
    # Test OpenRouter connection first
    if not test_openrouter_connection():
        print("Failed to connect to OpenRouter API. Please check your API key and connection.")
        return
    
    config = {"configurable": {"thread_id": "realtime_stock_demo"}}
    user_input = input("Enter a stock ticker (e.g., AAPL, TSLA, GOOGL): ")
    if not user_input:
        user_input = "What should I do with TSLA?"
    
    app = create_workflow()
    input_messages = [HumanMessage(content=user_input)]
    print(f"User: {user_input}")
    
    try:
        output = app.invoke({"messages": input_messages, "language": "English"}, config)
        print(f"Bot: {output['messages'][-1].content}")
    except Exception as e:
        print(f"Error running workflow: {str(e)}")

# ✅ DIRECT STOCK LOOKUP (BACKUP)
def direct_stock_lookup(ticker):
    try:
        print(f"Looking up {ticker}...")
        
        overview_response = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        overview = overview_response.json()
        
        quote_response = requests.get(
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        quote = quote_response.json().get("Global Quote", {})
        
        if not overview or "Symbol" not in overview or not quote or "05. price" not in quote:
            print(f"Could not fetch complete data for {ticker}.")
            return
        
        price = float(quote.get("05. price", "0"))
        change_percent = float(quote.get("10. change percent", "0").strip('%'))
        
        try:
            pe_ratio = float(overview.get("PERatio", "0"))
        except ValueError:
            pe_ratio = 0
            
        market_cap = overview.get("MarketCapitalization", "N/A")
        
        print(f"\n--- {ticker.upper()} Stock Data ---")
        print(f"Price: ${price:.2f}")
        print(f"Change: {change_percent}%")
        print(f"P/E Ratio: {pe_ratio}")
        print(f"Market Cap: {market_cap}")
        
    except Exception as e:
        print(f"Error in direct lookup: {str(e)}")

if __name__ == "__main__":
    # Offer options
    print("Choose an option:")
    print("1. Run full stock advisor")
    print("2. Test OpenRouter connection")
    print("3. Direct stock lookup")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        run_stock_example()
    elif choice == "2":
        if test_openrouter_connection():
            print("OpenRouter connection successful!")
        else:
            print("OpenRouter connection failed.")
    elif choice == "3":
        ticker = input("Enter stock ticker: ").upper()
        direct_stock_lookup(ticker)
    else:
        print("Invalid choice")