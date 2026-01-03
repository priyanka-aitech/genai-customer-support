'''from langchain_ollama import OllamaLLM
from langchain_classic.prompts import PromptTemplate
#from langchain.schema import Document
from langchain_core.documents import Document

from app.vector_store import get_retriever


# -------------------------
# LLM (Ollama)
# -------------------------
def get_llm():
    return OllamaLLM(
        model="llama3",
        temperature=0
    )


# -------------------------
# Prompt
# -------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful customer support assistant.

Use ONLY the context below to answer the question.
If the answer is not present in the context, say:
"I’m sorry, I don’t have that information."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -------------------------
# RAG Answer Function
# -------------------------
def get_rag_answer(question: str) -> str:
    retriever = get_retriever()

    # ✅ Correct way to retrieve documents (NO private methods)
    docs = retriever.invoke(question)

    if not docs:
        return "I’m sorry, I don’t have that information."

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = get_llm()
    prompt = PROMPT.format(context=context, question=question)

    response = llm.invoke(prompt)
    return response


# -------------------------
# Local Test
# -------------------------
if __name__ == "__main__":
    test_question = "How do I get a refund?"

    print("----- QUESTION -----")
    print(test_question)

    print("\n----- ANSWER -----")
    print(get_rag_answer(test_question))'''
#===================================================================================

from app.vector_store import get_retriever
from langchain_ollama import Ollama


def get_rag_answer(question: str) -> str:
    """
    Manual RAG pipeline:
    1. Retrieve relevant docs using Chroma retriever
    2. Build context
    3. Ask Ollama LLM
    """

    # 1️⃣ Get retriever (lazy-loaded)
    retriever = get_retriever()

    # 2️⃣ Retrieve documents (CORRECT API)
    docs = retriever.invoke(question)

    if not docs:
        return "Sorry, I could not find relevant information."

    # 3️⃣ Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    # 4️⃣ LLM
    llm = Ollama(
        model="llama3",
        temperature=0.2,
        num_ctx=1024
    )

    # 5️⃣ Prompt
    prompt = f"""
You are a customer support assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    # 6️⃣ Generate answer
    response = llm.invoke(prompt)

    return response


# Local test
if __name__ == "__main__":
    q = "How do I get a refund?"
    print("----- QUESTION -----")
    print(q)
    print("\n----- ANSWER -----")
    print(get_rag_answer(q))
