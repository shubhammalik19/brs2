import streamlit as st
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing_extensions import TypedDict
import os

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_NAME = os.getenv("DB_NAME", "BRS_DB")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Using embedding model: {EMBED_MODEL}")

# Initialize Embeddings & Vector Store
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embedding_model)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-2024-08-06", openai_api_key=OPENAI_API_KEY, temperature=0)

# Define Prompt Template
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

# ✅ Corrected State Definition
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieve function
def retrieve(state: dict):
    question = state.get("question", "").strip()
    if not question:
        return {"context": []}
    
    retrieved_docs = vector_store.similarity_search(question)
    return {"context": retrieved_docs}

# Chatbot function
def chatbot(state: dict):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    if not docs_content.strip():
        return {"answer": "I couldn't find relevant information."}

    formatted_prompt = prompt_template.format(
        question=state["question"], context=docs_content
    )

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

# Streamlit UI
st.title("🚀 RAG Chatbot - Company Assistant")
st.markdown("### Ask about your company, products, or services!")

user_input = st.text_input("Type your question here:")
if st.button("Ask AI"):
    if user_input.strip():
        response = graph.invoke({"question": user_input}, config)
        st.markdown("### AI Response:")
        st.success(response["answer"])
    else:
        st.warning("Please enter a valid question.")
