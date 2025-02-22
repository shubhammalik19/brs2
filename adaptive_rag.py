import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from pprint import pprint

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated import for ChromaDB
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # Updated Pydantic import
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Configuration from .env
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_NAME = os.getenv("DB_NAME", "BRS_DB")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

# Ensure the ChromaDB directory exists
os.makedirs(DB_NAME, exist_ok=True)

print(f"Using embedding model: {EMBED_MODEL}")

# Initialize Ollama Embeddings
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Initialize ChromaDB (Persistent Vector Store)
vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embedding_model)

# Initialize LLM (Using Llama 3.2)
llm = ChatOllama(model="llama3.2:latest", temperature=1)


# ** Graph State **
class GraphState(TypedDict):
    """Represents the state of the workflow graph."""
    question: str
    generation: str
    documents: List[str]


# ** Retrieval Function **
def retrieve_documents(state):
    """Retrieves relevant documents from ChromaDB."""
    question = state["question"].strip()
    if not question:
        return {"documents": [], "question": question}

    retrieved_docs = vector_store.similarity_search(question)
    print(f"Retrieved {len(retrieved_docs)} documents for the question: {question}")
    return {"documents": retrieved_docs, "question": question}


# ** Answer Generation Function **
def generate(state):
    """Generates an ERP assistant response based on retrieved documents."""
    context_text = "\n".join([doc.page_content for doc in state["documents"]])
    question_text = state["question"]

    # Define a system message to guide the assistant's behavior
    system_prompt = (
        "You are a highly knowledgeable ERP assistant designed to help customers and company staff. "
        "Your expertise includes our company's processes and software functionalities. "
        "Provide accurate and concise responses based on the available documents."
    )

    # Construct a better structured prompt
    prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser Question:\n{question_text}\n\nAnswer:"
    
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": llm.invoke(prompt)  # Pass a structured prompt
    }


# ** Document Grader Function **
class GradeDocuments(BaseModel):
    """Grades relevance of retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

retrieval_grader = ChatPromptTemplate.from_messages(
    [
        ("system", "Assess whether a retrieved document is relevant to a user question. "
                   "Grade it as 'yes' if relevant, otherwise 'no'."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
) | llm.with_structured_output(GradeDocuments)


def grade_documents(state):
    """Filters retrieved documents for relevance."""
    filtered_docs = [d for d in state["documents"] if retrieval_grader.invoke(
        {"question": state["question"], "document": d.page_content}).binary_score == "yes"]
    return {"documents": filtered_docs, "question": state["question"]}


# ** Workflow Graph Construction **
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    lambda state: "generate" if state["documents"] else END,
    {"generate": "generate"},
)
workflow.add_edge("generate", END)

# ** Set up memory for conversation history **
config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Ensure the graph image is saved every time the script runs
graph_image_path = "/home/shubham/BRS2/graph_image.png"

try:
    # Generate the Mermaid PNG image
    graph_image = app.get_graph().draw_mermaid_png()
    
    # Save the image
    with open(graph_image_path, "wb") as f:
        f.write(graph_image)

    print(f"✅ Graph image saved at: {graph_image_path}")

except Exception as e:
    print(f"⚠️ Error saving graph image: {e}")

# ** Chatbot Loop **
print("\n🤖 ERP Assistant is ready! Type 'exit' or 'quit' to end the chat.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Exiting the chat. Have a great day!")
        break

    inputs = {"question": user_input}
    response = None  # Ensure response is defined before using it

    for output in app.stream(inputs,config):
        response = output.get("generation", "I'm not sure how to answer that.")

    print(f"\nERP Assistant: {response}\n")
