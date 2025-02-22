from typing import Annotated, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing_extensions import TypedDict
from langchain import hub
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import shutil
import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_NAME = os.getenv("DB_NAME", "BRS_DB")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

print(f"Using embedding model: {EMBED_MODEL}")

# Initialize Ollama Embeddings
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Initialize ChromaDB
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embedding_model)

# Load prompt from LangChain Hub (ensure it exists, else use a fallback)
try:
    prompt = hub.pull("rlm/rag-prompt")
except Exception as e:
    print(f"Error loading prompt from LangChain Hub: {e}")
    prompt = lambda x: f"Q: {x['question']}\nContext: {x['context']}"

# Initialize LLM
llm = ChatOllama(model="deepseek-r1:latest")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieve function
def retrieve(state: State):
    question = state.get("question", "").strip()
    if not question:
        return {"context": []}
    
    retrieved_docs = vector_store.similarity_search(question)
    return {"context": retrieved_docs}

# Chatbot function
def chatbot(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Ensure context is not empty
    if not docs_content.strip():
        return {"answer": "I couldn't find relevant information."}
    
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Define Graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Stream updates
def stream_graph_updates(user_input: str):
    try:
        response = graph.invoke({"question": user_input})
        print(f"AI: {response['answer']}")
    except Exception as e:
        print(f"Error processing request: {e}")

# CLI Interaction
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
