from utils.config import OPENAI_API_KEY
#from openai import OpenAI

# Create OpenAI client
#client = OpenAI(api_key=OPENAI_API_KEY)

# Simple prompt
#response = client.chat.completions.create(
 #   model="gpt-4o-mini",
 #   messages=[
 #       {"role": "system", "content": "You are a helpful AI assistant."},
 #       {"role": "user", "content": "Explain transformers in one paragraph for a beginner."}
 #    ]
#)

# Print AI response
#print(response.choices[0].message.content)

# ----------------------------------------------------
# STEP 2: Load FAQ data from a text file
# ----------------------------------------------------

# We import Path from pathlib to safely handle file paths
# This avoids OS-specific path issues (Windows/Linux/Mac)
from pathlib import Path


# Define the path to the FAQ file
# __file__ refers to the current file (main.py)
# parent.parent moves us from app/ → project root
FAQ_FILE_PATH = Path(__file__).parent.parent / "data" / "raw_docs" / "faq.txt"


# Open the FAQ file in read mode ("r")
# encoding="utf-8" ensures text is read correctly
with open(FAQ_FILE_PATH, "r", encoding="utf-8") as file:
    faq_text = file.read()  # Read the entire file into a string


# Print the FAQ content to verify it loaded correctly
print("----- FAQ DATA LOADED SUCCESSFULLY -----")
print(faq_text)

# ----------------------------------------------------
# STEP 3: Split FAQ text into chunks
# ----------------------------------------------------

# Split the FAQ text into blocks using double newlines
# Each block represents one Question-Answer pair
faq_chunks = faq_text.split("\n\n")


# Remove any empty chunks (safety step)
faq_chunks = [chunk.strip() for chunk in faq_chunks if chunk.strip()]


# Print number of chunks created
print("\n----- FAQ CHUNKS CREATED -----")
print(f"Total chunks: {len(faq_chunks)}\n")


# Print each chunk with an index (for verification)
for index, chunk in enumerate(faq_chunks):
    print(f"Chunk {index + 1}:")
    print(chunk)
    print("-" * 40) 


# ----------------------------------------------------
# STEP 4: Create embeddings for FAQ chunks
# ⚠️ This step uses the OpenAI API
# ----------------------------------------------------

from openai import OpenAI  # OpenAI client library

# Create OpenAI client (uses API key from .env automatically)
client = OpenAI()


# This list will store embeddings along with their text
faq_embeddings = []


# Loop through each FAQ chunk and create its embedding
for chunk in faq_chunks:
    # Call the embedding API
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Cost-effective, production-grade embedding model
        input=chunk                       # Text to convert into numbers
    )

    # Extract the embedding vector from the response
    embedding_vector = response.data[0].embedding

    # Store both text and its embedding together
    faq_embeddings.append({
        "text": chunk,
        "embedding": embedding_vector
    })


# Print confirmation
print("\n----- EMBEDDINGS CREATED SUCCESSFULLY -----")
print(f"Total embeddings created: {len(faq_embeddings)}")
print(f"Embedding vector size: {len(faq_embeddings[0]['embedding'])}")

# ----------------------------------------------------
# STEP 5: Retrieve the most relevant FAQ chunk
# ⚠️ This step uses the OpenAI API for ONE embedding
# ----------------------------------------------------

import numpy as np  # Used for numerical calculations


def cosine_similarity(vector_a, vector_b):
    """
    Measures similarity between two vectors.
    Returns a value between -1 and 1.
    Higher = more similar.
    """
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


# Example user question (this simulates a real user query)
user_question = "How do I get a refund?"


# Convert the user question into an embedding
question_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_question
)

# Extract embedding vector for the question
question_embedding = question_response.data[0].embedding


# Find the most similar FAQ chunk
'''best_match = None
highest_similarity = -1


for item in faq_embeddings:
    similarity = cosine_similarity(
        question_embedding,
        item["embedding"]
    )
    print(f"Similarity with chunk: {similarity:.4f} | Text: {item['text']}")
    if similarity > highest_similarity:
        highest_similarity = similarity
        best_match = item["text"]
print(f"\nHighest similarity: {highest_similarity:.4f}")'''

# ------------------------------
# Step 5 Fix: Retrieve top N FAQ chunks
# ------------------------------
top_n = 3  # number of top chunks to retrieve

# Sort all FAQ embeddings by similarity
sorted_faqs = sorted(
    faq_embeddings,
    key=lambda x: cosine_similarity(question_embedding, x["embedding"]),
    reverse=True
)

# Take top N chunks
top_chunks = [x["text"] for x in sorted_faqs[:top_n]]

# Join top chunks as context for LLM
context_for_llm = "\n\n".join(top_chunks)

# Optional: print retrieved context
print("\n----- CONTEXT GIVEN TO LLM -----")
print(context_for_llm)


# Print retrieval result
print("\n----- RETRIEVAL RESULT -----")
print("User question:")
print(user_question)

print("\nMost relevant FAQ chunk:")
#print(best_match)
print(context_for_llm)


# ----------------------------------------------------
# STEP 6: Generate final answer using LLM (RAG)
# ⚠️ This step uses the OpenAI Chat API
# ----------------------------------------------------

# Construct a strict system prompt
# This tells the LLM HOW it is allowed to behave
system_prompt = """

You are a customer support assistant.

Use the provided context to answer the user's question.

If the answer is explicitly stated in the context, answer clearly.

If the answer is not explicitly stated but can be reasonably inferred
from the context, answer concisely using that inference.

If the question is completely unrelated to the context,
say exactly:
"I’m sorry, I don’t have that information."

"""

# Construct the user prompt
# We pass both the retrieved FAQ and the user question
'''user_prompt = f"""
Context:
{best_match}'''
#Fix
user_prompt = f"""
Context:
{context_for_llm}

Question:
{user_question}
"""

# Call the OpenAI Chat Completion API
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Cost-effective, strong reasoning model
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# Extract the assistant's reply
final_answer = response.choices[0].message.content

# Print the final answer
print("\n----- FINAL ANSWER -----")
print(final_answer)


