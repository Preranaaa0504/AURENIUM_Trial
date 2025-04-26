# FINANCIAL RISK ASSESSMENT CHATBOT

# Import required libraries
import os
from typing import Sequence
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing_extensions import Annotated, TypedDict


# Set LangSmith and DeepSeek API Keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

# Set DeepSeek API Key and Base URL (compatible with OpenAI API)
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize the model using DeepSeek-compatible model name
model = init_chat_model(
    model="openai/gpt-3.5-turbo",
    model_provider="openai"
)

# Define the state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Prompt tailored for financial risk assessment
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial risk assessment assistant. Assess the user's financial situation based on their responses. "
            "Ask relevant follow-up questions to evaluate risk tolerance, investment goals, age, income, and preferences. "
            "Then, provide investment advice and risk categorization (Low, Moderate, High). Speak in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Dummy token counter (approximation)
def dummy_token_counter(messages):
    return sum(len(msg.content.split()) // 0.75 for msg in messages)

# Trim old messages if tokens exceed limit
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=dummy_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Core model function
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages, "language": state.get("language", "English")})
    response = model.invoke(prompt)
    return {"messages": [response]}

# Optional: Risk category logic
def categorize_risk(state: State):
    user_input = state["messages"][-1].content.lower()
    if any(keyword in user_input for keyword in ["crypto", "startups", "high risk"]):
        category = "High Risk"
    elif any(keyword in user_input for keyword in ["mutual funds", "moderate", "balanced"]):
        category = "Moderate Risk"
    elif any(keyword in user_input for keyword in ["bonds", "safe", "low risk", "fixed deposit"]):
        category = "Low Risk"
    else:
        category = "Unclear – please provide more information."
    return {"messages": [AIMessage(content=f"Based on your input, your risk category is: **{category}**")]}

# Set up graph workflow
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_node("categorize", categorize_risk)
workflow.set_entry_point("model")
workflow.add_edge("model", "categorize")
workflow.set_finish_point("categorize")

# Set up memory saver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- Example Conversations ---

def run_financial_example():
    config = {"configurable": {"thread_id": "finance_thread_1"}}
    print("\n=== Financial Risk Assessment Example ===")

    # Initial user input
    query = "Hi, I’m 32, earning $90,000 annually, and want to invest for my child's education in 15 years."
    language = "English"
    input_messages = [HumanMessage(query)]
    print(f"Human: {query}")
    output = app.invoke({"messages": input_messages, "language": language}, config)
    print(f"AI: {output['messages'][-1].content}")

    # Follow-up input
    query = "I am okay with moderate risk. I prefer mutual funds over crypto."
    input_messages = [HumanMessage(query)]
    print(f"\nHuman: {query}")
    output = app.invoke({"messages": input_messages}, config)
    print(f"AI: {output['messages'][-1].content}")

def run_streaming_finance():
    config = {"configurable": {"thread_id": "finance_stream"}}
    print("\n=== Streaming Response: Financial Joke ===")
    query = "Tell me a funny investment joke."
    language = "English"
    input_messages = [HumanMessage(query)]
    print(f"Human: {query}")
    print("AI: ", end="")

    for chunk, metadata in app.stream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")

    print("\n")

# Run all examples
if __name__ == "__main__":
    print("Starting financial chatbot examples...\n")
    run_financial_example()
    run_streaming_finance()
    print("\nAll financial examples completed.")
