import json
import os
import pickle
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI()

CHUNKS_FILE = "docs/processed/chunks.json"
CHROMA_PATH = "docs/chroma_db"
BM25_FILE = "docs/processed/bm25_index.pkl"
BATCH_SIZE = 100

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chroma_collection(chunks):
    if os.getenv("CHROMA_HOST"):
        chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"),
                                            port=int(os.getenv("CHROMA_PORT", "8000")))
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="doc_chunks")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=[str(c["chunk_id"]) for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"], "strategy": c["strategy"], "entity": c["entity"]} for c in batch]
        )
        print(f"Embedded {i + len(batch)} / {len(chunks)} chunks")

    return collection

def tokenize(text):
    return text.lower().split()

def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25

if __name__ == "__main__":
    chunks = load_chunks()
    build_chroma_collection(chunks)
    build_bm25_index(chunks)
    print("Done: ChromaDB and BM25 index both built.")
