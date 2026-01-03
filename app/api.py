# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Import your working RAG function
#from app.rag_chain import get_rag_answer //commenting because lets not import rag_chain on the top level. We will lazily load it down.

app = FastAPI(
    title="GenAI Customer Support RAG API",
    version="1.0"
)

# Request model for Swagger / validation
class QuestionRequest(BaseModel):
    question: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------
# Lazy-loaded RAG function
# ---------
_rag_fn = None

def get_rag():
    global _rag_fn
    if _rag_fn is None:
        from app.rag_chain import get_rag_answer
        _rag_fn = get_rag_answer
    return _rag_fn


# RAG endpoint
'''@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Accepts a user question,
    retrieves relevant documents from the RAG pipeline,
    and returns the generated answer.
    """
    answer = get_rag_answer(request.question)
    return {"answer": answer}'''

# ---------
# Ask endpoint
# ---------
@app.post("/ask")
def ask(req: QuestionRequest):
    rag = get_rag()
    answer = rag(req.question)
    return {
        "question": req.question,
        "answer": answer
    }
