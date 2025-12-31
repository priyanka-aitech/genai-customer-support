# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel

# Import your working RAG function
from app.rag_chain import get_rag_answer

app = FastAPI(title="GenAI Customer Support RAG")

# Request model for Swagger / validation
class QuestionRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# RAG endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Accepts a user question,
    retrieves relevant documents from the RAG pipeline,
    and returns the generated answer.
    """
    answer = get_rag_answer(request.question)
    return {"answer": answer}
