import os
import requests
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.runnables import Runnable
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Set your API key for OpenRouter
os.environ["OPENROUTER_API_KEY"] = "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"

# Setup the LLM
llm = ChatOpenAI(
    model="openrouter/openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
)

# === Custom Alpha Vantage Stock Lookup Tool ===
ALPHA_VANTAGE_API_KEY = "MJY9U5NHJJ16ALMP" # Replace with your Alpha Vantage API key

def get_stock_price(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "Time Series (Daily)" in data:
            daily_data = data["Time Series (Daily)"]
            latest_day = list(daily_data.keys())[0]
            latest_data = daily_data[latest_day]
            return f"{ticker} - Open: {latest_data['1. open']}, Close: {latest_data['4. close']}, High: {latest_data['2. high']}, Low: {latest_data['3. low']}"
        else:
            return f"Error: Could not fetch data for {ticker}. {data.get('Note', 'Unknown error')}"
    else:
        return f"Error fetching data for {ticker}. HTTP {response.status_code}"

# Create a tool from the function
class AlphaVantageStockLookupTool:
    def __init__(self):
        self.name = "stock_lookup"
        self.description = "Look up stock prices by ticker symbol using Alpha Vantage API."
    
    def run(self, ticker):
        return get_stock_price(ticker)

# Instantiate the tool
alpha_vantage_tool = AlphaVantageStockLookupTool()

# === Financial Risk Document Tool ===
def create_risk_assessment_tool():
    loader = TextLoader("data/financial_risk.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    tool = create_retriever_tool(
        retriever,
        name="financial_risk_assessor",
        description="Get information and guidance on financial risk management concepts and strategies.",
    )
    return tool, qa_chain

risk_tool, risk_chain = create_risk_assessment_tool()

# === Agent with Tools ===
tools = [alpha_vantage_tool, risk_tool]

prompt = PromptTemplate.from_template("Answer the user's query precisely: {input}")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# === LangGraph ===
def stock_node(state):
    response = agent_executor.invoke({"input": state["messages"][-1]["content"]})
    state["messages"].append({"role": "assistant", "content": response})
    return state

graph = StateGraph(dict)
graph.add_node("stock_node", stock_node)
graph.set_entry_point("stock_node")
graph.set_finish_point("stock_node")
stock_graph: Runnable = graph.compile()

# === Stock Advisor Run Function ===
def run_stock_example():
    user_query = input("🧾 What would you like to know about stocks or financial risk? ")
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    result = stock_graph.invoke(inputs)
    for message in result["messages"]:
        if message["role"] == "assistant":
            print(f"\n🤖 {message['content']}")

# === OpenRouter Connection Test ===
def test_openrouter_connection():
    try:
        response = llm.invoke("Hello! Can you hear me?")
        print(f"\n🛰️ Response: {response.content}")
        return True
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

# === Direct Stock Info ===
def direct_stock_lookup(ticker):
    try:
        result = alpha_vantage_tool.run(ticker)
        print(f"\n📊 Stock Info for {ticker}:\n{result}")
    except Exception as e:
        print(f"\n⚠️ Couldn't fetch stock info for {ticker}: {e}")

# === Friendly Chatbot Interface ===
if __name__ == "__main__":
    print("👋 Hey there! I'm your smart Financial Advisor Bot.")
    print("Here’s what I can do for you today:")
    print("💬 Just type something like:")
    print("  - 'Start the stock advisor'")
    print("  - 'Check OpenRouter connection'")
    print("  - 'Look up stock info for TSLA'")

    user_input = input("\n🧠 What would you like to do? ").lower()

    if "advisor" in user_input:
        run_stock_example()
    elif "connection" in user_input or "openrouter" in user_input:
        if test_openrouter_connection():
            print("\n✅ Connection is good!")
        else:
            print("\n❌ Connection issue detected.")
    elif "lookup" in user_input or "stock" in user_input:
        ticker = input("📌 Enter the stock ticker (like AAPL, MSFT): ").upper()
        direct_stock_lookup(ticker)
    else:
        print("🤔 Hmm, I didn’t catch that. Try saying something like 'start the advisor' or 'check connection'.")
