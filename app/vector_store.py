'''# app/vector_store.py

"""
This file is responsible for:
1. Reading FAQ data
2. Creating embeddings
3. Storing embeddings in Chroma vector database
4. Returning a retriever for semantic search

NO LLM logic should exist here.
"""

# ----------------------------
# Standard library imports
# ----------------------------
from pathlib import Path  # For safe file path handling

# ----------------------------
# LangChain imports
# ----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTORSTORE = None
# ----------------------------
# CONSTANTS
# ----------------------------

# Location of FAQ file
FAQ_PATH = Path("data/raw_docs/faq.txt")

# Directory where Chroma will store embeddings on disk
CHROMA_DB_DIR = "chroma_db"


# ----------------------------
# STEP 1: Load FAQ text
# ----------------------------

def load_faq_text():
    """
    Reads the FAQ text file and returns it as a string.
    """

    if not FAQ_PATH.exists():
        raise FileNotFoundError("FAQ file not found at data/raw/faq.txt")

    return FAQ_PATH.read_text(encoding="utf-8")


# ----------------------------
# STEP 2: Split FAQ into chunks
# ----------------------------

def split_into_chunks(text):
    """
    Splits long FAQ text into smaller chunks.
    This improves embedding quality and retrieval accuracy.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        # Max characters per chunk
        chunk_overlap=50       # Overlap to preserve context
    )

    return splitter.split_text(text)


# ----------------------------
# STEP 3: Create Vector Store
# ----------------------------

def create_vector_store():
    """
    Creates or loads a Chroma vector store with embedded FAQ chunks.
    """

    # Load raw FAQ text
    faq_text = load_faq_text()

    # Split FAQ into chunks
    chunks = split_into_chunks(faq_text)

    # Create embedding model
    # We use a small, fast, free HuggingFace model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create or load Chroma DB
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    return vectorstore


# ----------------------------
# STEP 4: Expose Retriever
# ----------------------------

def get_retriever():
    """
    Returns a retriever object that can be used
    to fetch relevant FAQ chunks for a query.
    """

    vectorstore = create_vector_store()

    # k = number of top relevant chunks to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 3})'''

#=============================================================================================
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

VECTORSTORE = None

def load_faq_text():
    path = "data/raw_docs/faq.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAQ file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_vector_store():
    global VECTORSTORE
    if VECTORSTORE is not None:
        return VECTORSTORE

    faq_text = load_faq_text()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(faq_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")
    vectordb.persist()
    VECTORSTORE = vectordb
    return VECTORSTORE

def get_retriever():
    return create_vector_store().as_retriever()

