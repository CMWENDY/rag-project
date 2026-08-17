import json
import os
import pickle
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder


load_dotenv()
client = OpenAI()

CHUNKS_FILE = "docs/processed/chunks.json"
CHROMA_PATH = "docs/chroma_db"
BM25_FILE = "docs/processed/bm25_index.pkl"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

if os.getenv("CHROMA_HOST"):
    chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"),
                                        port=int(os.getenv("CHROMA_PORT", "8000")))
else:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="doc_chunks")

with open(BM25_FILE, "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]

chunk_lookup = {str(c["chunk_id"]): c for c in chunks}

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def tokenize(text):
    return text.lower().split()

def build_where(entity=None, strategy=None):
    conditions = []
    if entity is not None:
        conditions.append({"entity": entity})
    if strategy is not None:
        conditions.append({"strategy": strategy})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def dense_search(query, top_k=10, entity=None, strategy=None):
    query_embedding = embed_text(query)

    where = build_where(entity, strategy)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where
    )
    return results["ids"][0]

def sparse_search(query, top_k=10, entity=None, strategy=None):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    result_ids = []
    for i in ranked_indices:
        if entity is not None and chunks[i]["entity"] != entity:
            continue
        if strategy is not None and chunks[i]["strategy"] != strategy:
            continue
        result_ids.append(str(chunks[i]["chunk_id"]))
        if len(result_ids) == top_k:
            break
    return result_ids

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}

    for rank, chunk_id in enumerate(dense_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, chunk_id in enumerate(sparse_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return ranked_ids

def rerank(query, candidates, top_k=5):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored_candidates[:top_k]]


def hybrid_search(query, top_k=5, rerank_pool=20, entity=None, strategy=None):
    dense_results = dense_search(query, top_k=10, entity=entity, strategy=strategy)
    sparse_results = sparse_search(query, top_k=10, entity=entity, strategy=strategy)
    ranked_ids = reciprocal_rank_fusion(dense_results, sparse_results)

    candidates = [chunk_lookup[cid] for cid in ranked_ids[:rerank_pool]]
    return rerank(query, candidates, top_k=top_k)

def refresh():
    """Reload the BM25 index and chunk lookup from disk (after new docs are added)."""
    global bm25, chunks, chunk_lookup, collection
    with open(BM25_FILE, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]
    chunk_lookup = {str(c["chunk_id"]): c for c in chunks}
    collection = chroma_client.get_or_create_collection(name="doc_chunks")


if __name__ == "__main__":
    query = "what is the process for offboarding an employee"
    results = hybrid_search(query)
    for r in results:
        print(f"[{r['source']}] {r['text'][:150]}...")
