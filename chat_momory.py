from typing import Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
import sys
from langgraph.checkpoint.memory import MemorySaver

# ✅ Define chatbot state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ✅ Initialize Graph
graph_builder = StateGraph(State)

# ✅ Define the tool correctly using @tool
@tool
def sum_of_2_numbers(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b

# ✅ Use a model that supports tools OR fallback
try:
    llm = ChatOllama(model="llama3.2:latest",temperature=1)  # ✅ Change to "llama3:latest"
    llm_with_tools = llm.bind_tools([sum_of_2_numbers])  # ✅ Fix binding
except Exception as e:
    print(f"Warning: The model does not support tools. Error: {e}")
    llm_with_tools = llm  # Fallback to normal LLM

# ✅ Chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ✅ Add chatbot node
graph_builder.add_node("chatbot", chatbot)

# ✅ Define tool node correctly
tool_node = ToolNode(tools=[sum_of_2_numbers])
graph_builder.add_node("tools", tool_node)

# ✅ Define conditional execution
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# ✅ Set entry point
graph_builder.set_entry_point("chatbot")
config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

""" try:
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("/home/shubham/BRS2/graph_image.png", "wb") as f:
        f.write(graph_image)
except Exception:
    # This requires some extra dependencies and is optional
    pass
graph.invoke({"messages": [HumanMessage(content="What is the sum of 2 and 3?")]})  # ✅ Fix message format
sys.exit(0); """
# ✅ Stream updates with error handling
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
