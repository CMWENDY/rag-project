from retriever import hybrid_search
from generator import generate_answer
from verifier import verify_citations
from scorer import confidence_score

COMPLETENESS_THRESHOLD = 0.3

def unknown_response(query, chunks, retrieval):
    closest_documents = sorted({c["source"] for c in chunks})
    return {
        "status": "insufficient_confidence",
        "message": (
            f"I couldn't find information I'm confident answers your question: "
            f"\"{query}\". The retrieved material wasn't a strong enough match, "
            f"so I'm not going to guess."
        ),
        "retrieval_confidence": round(retrieval, 2),
        "closest_documents": closest_documents,
        "suggestion": "These documents were the closest matches and may be worth "
                      "checking manually.",
        "chunks": chunks,
    }

def answer_question(query, top_k=15, entity=None, strategy=None):
    chunks = hybrid_search(query, top_k=top_k, entity=entity, strategy=strategy)

    if not chunks:                        # nothing retrieved at all -> can't answer
        return unknown_response(query, chunks, 0.0)

    answer, _ = generate_answer(query, top_k=top_k, entity=entity,
                                strategy=strategy, chunks=chunks)
    verification = verify_citations(answer, chunks)
    scores = confidence_score(query, answer, chunks, verification)

    # POST-generation gate: if the answer didn't actually answer, decline
    if scores["answer_completeness"] < COMPLETENESS_THRESHOLD:
        return unknown_response(query, chunks, scores["retrieval_confidence"])

    return {
        "status": "answered",
        "answer": answer,
        "confidence": scores,
        "sources": sorted({c["source"] for c in chunks}),
        "chunks": chunks,
    }


if __name__ == "__main__":
    for q in [
        "what dental benefits does GitLab offer US employees",
        "what is GitLab's policy on submarine leave",
    ]:
        print("=" * 60)
        print("QUERY:", q)
        result = answer_question(q, entity="us", strategy="structure", top_k=15)
        print("STATUS:", result["status"])
        if result["status"] == "answered":
            print(result["answer"])
            print("confidence:", result["confidence"]["composite"])
            print("full breakdown:", result["confidence"])
        else:
            print(result["message"])
            print("closest docs:", result["closest_documents"])
