import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from retriever import reranker

load_dotenv()
client = OpenAI()

COMPLETENESS_PROMPT = """You evaluate whether an ANSWER fully addresses a QUESTION.
The question may have several parts. Estimate what fraction of the question's
parts the answer actually addresses.

Respond ONLY as JSON in this exact shape:
{"completeness": <number between 0 and 1>, "reason": "<one short sentence>"}
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def retrieval_confidence(query, chunks, top_n=5):
    if not chunks:
        return 0.0
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)
    relevances = sorted((sigmoid(s) for s in scores), reverse=True)[:top_n]
    return sum(relevances) / len(relevances)

def citation_coverage(verification_results):
    if not verification_results:
        return 0.0
    supported = sum(1 for r in verification_results if r["supported"])
    return supported / len(verification_results)

def answer_completeness(query, answer):
    user_prompt = f"QUESTION: {query}\n\nANSWER: {answer}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": COMPLETENESS_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def confidence_score(query, answer, chunks, verification_results):
    retrieval = retrieval_confidence(query, chunks)
    coverage = citation_coverage(verification_results)
    completeness = answer_completeness(query, answer)["completeness"]

    composite = (0.4 * retrieval) + (0.3 * coverage) + (0.3 * completeness)

    return {
        "composite": round(float(composite), 2),
        "retrieval_confidence": round(float(retrieval), 2),
        "citation_coverage": round(float(coverage), 2),
        "answer_completeness": round(float(completeness), 2),
    }

if __name__ == "__main__":
    from generator import generate_answer
    from verifier import verify_citations

    query = "what medical and dental benefits does GitLab offer US employees"
    answer, chunks = generate_answer(query)
    verification_results = verify_citations(answer, chunks)

    scores = confidence_score(query, answer, chunks, verification_results)

    print("ANSWER:\n", answer, "\n")
    print("CONFIDENCE:")
    for name, value in scores.items():
        print(f"  {name}: {value}")
