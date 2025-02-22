import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
# Load environment variables from .env
load_dotenv()

# Ensure API key is set for Tavily
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY is not set. Check your .env file or environment variables.")

# Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# Initialize Tavily and Ollama
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatOllama(model="llama3.2:latest", temperature=1)  # ✅ Changed to Llama 3.2

llm_with_tools = llm.bind_tools(tools)

# Chatbot function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Build LangGraph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    try:
        response = graph.invoke({"messages": [HumanMessage(content=user_input)]},config)
        print(f"AI: {response['messages'][-1].content}")
    except Exception as e:
        print(f"Error processing request: {e}")


# ✅ CLI Interaction
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
