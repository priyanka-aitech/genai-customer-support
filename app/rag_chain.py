from langchain_ollama import OllamaLLM
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
    print(get_rag_answer(test_question))
