# -------------------------------
# STEP 0: Basic imports
# -------------------------------

# Used to build file paths safely (Windows/Mac/Linux)
import os

# Load environment variables from .env
from dotenv import load_dotenv

# -------------------------------
# STEP 1: LangChain components
# -------------------------------

# Loads text files into LangChain "Document" objects
from langchain_community.document_loaders import TextLoader

# Splits large text into smaller overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Converts text into embeddings using OpenAI
from langchain_openai import OpenAIEmbeddings

# Chat model wrapper (LLM)
from langchain_openai import ChatOpenAI

# Vector store for similarity search (in-memory)
from langchain_community.vectorstores import FAISS

# Prompt template utility
from langchain_core.prompts import ChatPromptTemplate

# Chain utility to connect retriever + LLM
#from langchain.chains.retrieval_qa.base import RetrievalQA //always threw error. Hence adding the following 2 lines.

# Creates a chain that combines retrieved documents
#from langchain.chains.combine_documents import create_stuff_documents_chain

# Creates the retrieval + generation pipeline
#from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# STEP 1: Load environment variables
# ============================================================

# Loads OPENAI_API_KEY from .env
load_dotenv()


# ============================================================
# STEP 2: Load FAQ data
# ============================================================

FAQ_PATH = "data/raw_docs/faq.txt"

# Reads the FAQ file and converts it into LangChain Documents
loader = TextLoader(FAQ_PATH)
documents = loader.load()


# ============================================================
# STEP 3: Split FAQ into chunks
# ============================================================

# Chunking ensures text fits into embedding model limits
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # size of each chunk
    chunk_overlap=100      # overlap preserves context
)

docs = text_splitter.split_documents(documents)


# ============================================================
# STEP 4: Create embeddings
# ============================================================

# Converts text into numerical vectors
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ============================================================
# STEP 5: Create vector store (FAISS)
# ============================================================

# Stores embeddings and enables fast similarity search
vectorstore = FAISS.from_documents(docs, embeddings)


# ============================================================
# STEP 6: Create retriever
# ============================================================

# Retriever finds the most relevant chunks for a query
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # top 3 chunks
)


# ============================================================
# STEP 7: Create LLM
# ============================================================

# Chat model used to generate answers
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# STEP 8: Define prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a customer support assistant.

Use the context below to answer the question. Give relevant info.

Context:
{context}

Question:
{question}
"""
)


# ============================================================
# STEP 9: Build RAG pipeline (NEW Runnable style)
# ============================================================

# Formats retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG pipeline:
# 1. Retrieve documents
# 2. Format them
# 3. Inject into prompt
# 4. Call LLM
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
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
print(response.content)