from openai import OpenAI
from dotenv import load_dotenv
from retriever import hybrid_search

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Only use information found in the numbered context blocks below.
- Every claim you make must be followed by a bracketed citation, like [1] or [2], pointing to the context block that supports it.
- If the context does not contain enough information to answer the question, say so clearly instead of guessing.
- Do not use any outside knowledge, even if you know the answer.
"""

def build_context_blocks(chunks):
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (Source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(blocks)

def build_user_prompt(query, chunks):
    context = build_context_blocks(chunks)
    return f"""Context:
{context}

Question: {query}

Answer the question using only the context above. Cite your sources using [1], [2], etc.
"""

def generate_answer(query, top_k=10, entity=None, strategy=None, chunks=None):
    if chunks is None:
        chunks = hybrid_search(query, top_k=top_k, entity=entity, strategy=strategy)
    user_prompt = build_user_prompt(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content, chunks

if __name__ == "__main__":
    query = "what medical, dental, and retirement benefits does GitLab offer US employees"
    answer, chunks = generate_answer(query, entity="us")

    print("ANSWER:\n", answer)
    print("\nSOURCES:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] {chunk['source']}")
