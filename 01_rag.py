from typing import Annotated, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing_extensions import TypedDict
from langchain import hub
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_NAME = os.getenv("DB_NAME", "BRS_DB")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Using embedding model: {EMBED_MODEL}")

# Initialize Ollama Embeddings
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Initialize ChromaDB
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embedding_model)

# Load prompt from LangChain Hub (ensure it exists, else use a fallback)
prompt_template = """You are an AI assistant that helps customers, company staff, and clients understand the company's products and services. 

**Your Role:**
- Assist users in understanding the company's operations and offerings.
- Provide clear, structured responses to questions.
- Guide users in effectively using the company’s products.

**User Query:** {question}

**Company Context:** {context}

**Response Format:**
- **If explaining a product/service**, provide:
  - **What it does**
  - **Who it is for**
  - **How to use it (step-by-step, if needed)**

- **If answering a general company-related query**, structure the response as:
  - **Brief overview**
  - **Key details relevant to the question**
  - **Next steps or recommendations**

Now, based on the above instructions, generate a clear and informative response to the user's question.
"""


# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

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

    if not docs_content.strip():
        return {"answer": "I couldn't find relevant information."}

    formatted_prompt = prompt_template.format(
        question=state["question"], context=docs_content
    )

    # OpenAI LLM expects messages format
    response = llm.invoke([{"role": "system", "content": formatted_prompt}])

    return {"answer": response.content}

# Define Graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "chatbot")
graph_builder.add_edge("chatbot", END)

config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = graph_builder.compile()

# Stream updates
def stream_graph_updates(user_input: str):
    try:
        response = graph.invoke({"question": user_input}, config)
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
