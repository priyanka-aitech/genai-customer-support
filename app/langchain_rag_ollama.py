# ============================================================
# RAG with Ollama LLM + Chroma vectorstore
# ============================================================

# ----------- Basic imports -----------
import os
from dotenv import load_dotenv

# ----------- LangChain imports (newest stable APIs) -----------
from langchain_community.document_loaders import TextLoader  # Load text files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split text into chunks

# Vectorstore
from langchain_community.vectorstores import Chroma  # Chroma instead of FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Use HF embeddings for local LLM

# Ollama LLM
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM

# Prompt utilities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# STEP 1: Load environment variables
# ============================================================
load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Model name for Ollama (local)
CHROMA_PERSIST_DIR = "chroma_db"  # Folder to store Chroma embeddings

# ============================================================
# STEP 2: Load FAQ data
# ============================================================
FAQ_PATH = "data/raw_docs/faq.txt"
loader = TextLoader(FAQ_PATH)  # Read text file
documents = loader.load()  # Returns list of LangChain Documents

# ============================================================
# STEP 3: Split FAQ into chunks
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Max tokens per chunk
    chunk_overlap=100    # Overlap ensures context is preserved
)
docs = text_splitter.split_documents(documents)

# ============================================================
# STEP 4: Create embeddings
# ============================================================
# Use HuggingFace embeddings (local, free) instead of OpenAI
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# STEP 5: Create vector store with Chroma
# ============================================================
# Persist allows you to save embeddings locally
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
vectorstore.persist()  # Save embeddings to disk

# ============================================================
# STEP 6: Create retriever
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
)

# ============================================================
# STEP 7: Create Ollama LLM
# ============================================================
llm = OllamaLLM(
    model=OLLAMA_MODEL,  # Ollama local model
    temperature=0
)

# ============================================================
# STEP 8: Define prompt
# ============================================================
prompt = ChatPromptTemplate.from_template(
    """
You are a customer support assistant.

Use the context below to answer the user's question. Give relevant info.

Context:
{context}

Question:
{question}
"""
)

# ============================================================
# STEP 9: Build RAG pipeline (Runnable style)
# ============================================================
# Function to format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG pipeline:
rag_chain = (
    {
        "context": retriever | format_docs,  # retrieve & format
        "question": RunnablePassthrough()    # pass user question
    }
    | prompt
    | llm
)

# ============================================================
# STEP 10: Ask a question
# ============================================================
query = "How do I get a refund?"

response = rag_chain.invoke(query)

print("\n----- QUESTION -----")
print(query)

print("\n----- ANSWER -----")
print(response)
